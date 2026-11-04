"""Universal test hermetic isolation guard.

Ensures that whenever tests are run under ANY test runner (pytest, unittest,
or direct invocation), HERMES_HOME is guaranteed to be redirected to a temporary
directory, and never touches ~/.hermes or production state on the host.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

_CLEANUP_PATHS = []


def _cleanup_temp_dirs():
    for p in _CLEANUP_PATHS:
        try:
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


atexit.register(_cleanup_temp_dirs)


def ensure_hermetic_test_environment() -> str:
    """Guarantee that HERMES_HOME points to an isolated sandbox."""
    if os.environ.get("HERMES_TEST_ISOLATION"):
        return os.environ.get("HERMES_HOME", "")

    real_root = None
    try:
        real_root = (Path(os.path.expanduser("~")) / ".hermes").resolve()
    except Exception:
        pass

    current_home = os.environ.get("HERMES_HOME", "").strip()
    is_safe = False
    if current_home:
        try:
            resolved = Path(current_home).expanduser().resolve()
            if real_root is None or (resolved != real_root and real_root not in resolved.parents):
                is_safe = True
        except Exception:
            is_safe = False

    if not is_safe:
        sandbox_dir = tempfile.mkdtemp(prefix="hermes_unittest_sandbox_")
        _CLEANUP_PATHS.append(sandbox_dir)
        os.environ["HERMES_HOME"] = sandbox_dir
        # Create minimal config so tools loading config don't crash on missing file
        try:
            (Path(sandbox_dir) / "config.yaml").write_text("{}\n", encoding="utf-8")
        except Exception:
            pass

    os.environ["HERMES_TEST_ISOLATION"] = "1"
    return os.environ.get("HERMES_HOME", "")


# Auto-apply on import
ensure_hermetic_test_environment()
