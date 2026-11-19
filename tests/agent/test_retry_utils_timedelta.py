"""Unit tests for agent.retry_utils.coerce_retry_after_seconds."""

from datetime import timedelta
import math

import pytest

from agent.retry_utils import coerce_retry_after_seconds


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(seconds=1488), 1488.0),
        (timedelta(seconds=1486), 1486.0),
        (timedelta(seconds=0), 0.0),
        (timedelta(days=1, seconds=5), 86405.0),
        (3, 3.0),
        (0.2, 0.2),
        (0, 0.0),
        ("30", 30.0),
        ("2.5", 2.5),
        ("0:24:46", 1486.0),
        ("1:02:03.5", 3723.5),
        ("Retry in 0:24:46", 1486.0),
        ("Flood control exceeded. Retry in 0:24:46", 1486.0),
        ("Retry in 30 seconds", 30.0),
        ("Retry after 12.5s", 12.5),
        ("1 day, 0:00:05", 86405.0),
    ],
)
def test_coerce_retry_after_seconds_valid(value, expected):
    result = coerce_retry_after_seconds(value)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        -1,
        -0.5,
        "-1",
        timedelta(seconds=-1),
        "Retry in -5",
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "inf",
        "-inf",
        "",
        "   ",
        "not a duration",
        "0:65:00",
        object(),
        [],
        {},
    ],
)
def test_coerce_retry_after_seconds_invalid_returns_none(value):
    result = coerce_retry_after_seconds(value)
    assert result is None
    assert not (isinstance(result, float) and math.isnan(result))
