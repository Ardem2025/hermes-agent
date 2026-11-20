"""An ``unknown`` turn nobody is waiting on settles as interrupted (#134889 GW-10, #139532).

A turn lost across an owner restart is fenced ``unknown`` and pauses its session's FIFO, because
only the person who sent it knows whether to resend it. An automation delivery has no such person:
a cron run (``CRON_PRINCIPAL_ID``), a process completion or heartbeat (``automation:<route>``,
``local_automation_v1``, a native envelope's ``automation`` block) or a webhook delivery
(``webhook_delivery``) or a Kanban card attempt (``kanban_card_v1``, set by the dispatcher). Pausing on it stalls every message queued behind it until an operator
notices (a Bot Chat sat ten hours behind one).

Such a row settles through the same fenced resolution a person's Discard uses
(``resolve_unknown_session_input``): terminal ``interrupted``, never requeued or replayed, with a
Hermes transcript notice committed in the same transaction. No second recovery store.
"""
from __future__ import annotations

import logging

from hermes_state_keys import CRON_PRINCIPAL_ID

logger = logging.getLogger(__name__)


def is_unattended(row) -> bool:
    """True for an admission no person submitted (so nobody can be asked to resend it)."""
    principal = row.get('principal_id') or ''
    if principal == CRON_PRINCIPAL_ID or principal.startswith('automation:'):
        return True
    payload = row.get('payload') or {}
    native = payload.get('native_text_v1') or {}
    # A webhook delivery: the provider already got its 202 and nobody can be asked to resend it.
    # A Kanban card attempt (#134889 KB-8): the dispatcher's marker, committed only by
    # ``SessionAuthority.submit(_kanban_card=...)``; a card row main minted without it still pauses.
    return ('local_automation_v1' in payload or 'kanban_card_v1' in payload
            or 'automation' in native or 'webhook_delivery' in native)


def _unattended_unknown(rows):
    return [row for row in rows if row['status'] == 'unknown' and is_unattended(row)]


def settle_recovered_unattended(db, epoch) -> int:
    """Boot recovery, right after ``recover_session_inputs`` fenced lost turns ``unknown`` and before
    any FIFO is scheduled: settle every unattended one. Synchronous; runs off the owner loop."""
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from gateway.session_ingress_media import release_admission_media
    from gateway.session_results import close_discarded_turn
    from hermes_state_runtime import RuntimeStoreError, _row, resolve_unknown_session_input
    with db._read_ctx() as conn:
        rows = [_row(r) for r in conn.execute(
            "SELECT * FROM session_admissions WHERE status='unknown' ORDER BY seq").fetchall()]
    settled = 0
    for row in _unattended_unknown(rows):
        try:
            resolve_unknown_session_input(
                db, epoch=epoch, admission_id=row['admission_id'], generation=row['generation'],
                _terminal_write=lambda conn, lost: close_discarded_turn(db, conn, lost, AUTOMATION_INTERRUPTED_NOTICE))
        except RuntimeStoreError as exc:
            logger.warning('Lost automation admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        release_admission_media(db, row['admission_id'])
        settled += 1
        logger.info('Lost automation admission %s on session %s settled interrupted (not replayed)',
                    row['admission_id'], row['target_session_id'])
    return settled


async def settle_unattended_unknown(authority, ref) -> bool:
    """The drain found an ``unknown`` head (fenced in this owner, e.g. a lost managed worker):
    settle the unattended ones through the owner's resolution. True when any settled, so the
    drain re-reads its FIFO instead of pausing; a person's ``unknown`` row still pauses it."""
    import asyncio

    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=ref.session_id)
    settled = False
    for row in _unattended_unknown(rows):
        try:
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'],
                                                     notice=AUTOMATION_INTERRUPTED_NOTICE)
        except RuntimeStoreError as exc:
            logger.warning('Lost automation admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        settled = True
    return settled


def is_reset_reconcilable_native(row) -> bool:
    """True for a messaging ``native_text_v1`` admission (``unknown`` or ``queued``) eligible for
    reconciliation when its conversation ended via ``session_reset``."""
    if row.get('status') not in ('unknown', 'queued'):
        return False
    principal = row.get('principal_id') or ''
    if not principal.startswith('messaging:'):
        return False
    payload = row.get('payload')
    if not isinstance(payload, dict) or 'native_text_v1' not in payload:
        return False
    return not is_unattended(row)


def _is_session_reset_ended(conn, session_id: str) -> bool:
    """True when *session_id* (or its compression lineage tip) ended with ``session_reset``."""
    from hermes_state_runtime import RuntimeStoreError, _canonical_chain
    row = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        return False
    if row[0] is not None and row[1] == 'session_reset':
        return True
    try:
        tip_id = _canonical_chain(conn, session_id)[-1]
    except RuntimeStoreError:
        return False
    if tip_id == session_id:
        return False
    tip = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id=?", (tip_id,)).fetchone()
    return bool(tip is not None and tip[0] is not None and tip[1] == 'session_reset')


def settle_reset_native_admissions(db, epoch: int) -> int:
    """Boot recovery: close orphaned ``unknown``/``queued`` ``native_text_v1`` admissions from
    prior epochs whose conversation already ended with ``session_reset``."""
    from gateway.session_ingress_media import release_admission_media
    from gateway.session_results import close_discarded_turn
    from hermes_state_runtime import (
        RuntimeStoreError, _row, cancel_session_input, resolve_unknown_session_input,
    )
    with db._read_ctx() as conn:
        raw_rows = conn.execute(
            "SELECT * FROM session_admissions WHERE status IN ('unknown', 'queued') "
            "AND (owner_epoch IS NULL OR owner_epoch < ?) ORDER BY seq",
            (epoch,),
        ).fetchall()
        candidates = []
        reset_cache: dict[str, bool] = {}
        for raw in raw_rows:
            row = _row(raw)
            if not is_reset_reconcilable_native(row):
                continue
            sid = row['target_session_id']
            if sid not in reset_cache:
                reset_cache[sid] = _is_session_reset_ended(conn, sid)
            if reset_cache[sid]:
                candidates.append(row)
    settled = 0
    for row in candidates:
        aid = row['admission_id']
        sid = row['target_session_id']
        if row['status'] == 'unknown':
            try:
                resolve_unknown_session_input(
                    db, epoch=epoch, admission_id=aid, generation=row['generation'],
                    _terminal_write=lambda conn, lost: close_discarded_turn(db, conn, lost, None))
            except RuntimeStoreError as exc:
                logger.warning('Orphaned reset unknown admission %s on %s left unknown: %s', aid, sid, exc.reason)
                continue
            release_admission_media(db, aid)
            settled += 1
            logger.info('Orphaned reset admission %s on session %s settled interrupted', aid, sid)
        elif row['status'] == 'queued':
            try:
                cancel_session_input(db, epoch=epoch, admission_id=aid)
            except RuntimeStoreError as exc:
                logger.warning('Orphaned reset queued admission %s on %s left queued: %s', aid, sid, exc.reason)
                continue
            release_admission_media(db, aid)
            settled += 1
            logger.info('Orphaned reset admission %s on session %s settled cancelled', aid, sid)
    if settled:
        logger.info('Settled %d orphaned reset native admission(s)', settled)
    return settled
