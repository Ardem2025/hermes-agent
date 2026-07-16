"""Regression coverage for fail-closed Telegram role-topic prompts."""

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_state import AsyncSessionDB, SessionDB


async def _role_prompt(db: SessionDB, source: SessionSource) -> str:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_db = AsyncSessionDB(db)
    return await runner._telegram_topic_role_prompt(source)


def _source(*, thread_id: str | None = "17585") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        chat_type="dm",
        user_id="208214988",
        thread_id=thread_id,
    )


@pytest.mark.asyncio
async def test_life_coach_role_prompt_uses_actual_async_session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="life-coach", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:17585",
        session_id="life-coach",
        role_id="life-coach",
        prompt_revision="v1",
    )

    prompt = await _role_prompt(db, _source())

    assert "[ROLE TOPIC: life-coach/v1]" in prompt
    assert "onboarding questionnaire" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role_id", "revision"),
    [(None, None), ("life-coach", "v2"), ("other-role", "v1")],
)
async def test_role_prompt_fails_closed_for_unrecognized_binding(tmp_path, role_id, revision):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="other", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:17585",
        session_id="other",
        role_id=role_id,
        prompt_revision=revision,
    )

    assert await _role_prompt(db, _source()) == ""


@pytest.mark.asyncio
async def test_role_prompt_requires_telegram_topic_and_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    assert await _role_prompt(db, _source(thread_id=None)) == ""

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_db = None
    assert await runner._telegram_topic_role_prompt(_source()) == ""
