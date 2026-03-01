"""Tests for the retry utilities."""

import time
import pytest
from unittest.mock import patch, MagicMock
from personal_assistant.tools.retry_utils import retry_with_backoff


class TestRetryWithBackoff:
    def test_success_on_first_attempt(self):
        calls = []

        @retry_with_backoff(max_retries=3)
        def succeed():
            calls.append(1)
            return "ok"

        assert succeed() == "ok"
        assert len(calls) == 1

    def test_retries_on_rate_limit_keyword(self):
        calls = []

        @retry_with_backoff(max_retries=2, initial_delay=0.01, retryable_exceptions=(Exception,))
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise Exception("quota exceeded - too many requests")
            return "done"

        with patch("personal_assistant.tools.retry_utils.time.sleep"):
            result = flaky()

        assert result == "done"
        assert len(calls) == 3

    def test_raises_after_max_retries(self):
        calls = []

        @retry_with_backoff(max_retries=2, initial_delay=0.01, retryable_exceptions=(Exception,))
        def always_fail():
            calls.append(1)
            raise Exception("429 resource exhausted")

        with patch("personal_assistant.tools.retry_utils.time.sleep"):
            with pytest.raises(Exception, match="resource exhausted"):
                always_fail()

        assert len(calls) == 3  # 1 initial + 2 retries

    def test_non_retryable_exception_raises_immediately(self):
        calls = []

        @retry_with_backoff(max_retries=3, initial_delay=0.01, retryable_exceptions=(ValueError,))
        def wrong_type():
            calls.append(1)
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            wrong_type()

        assert len(calls) == 1

    def test_delay_does_not_exceed_max(self):
        sleep_calls = []

        @retry_with_backoff(
            max_retries=3,
            initial_delay=50.0,
            backoff_factor=2.0,
            max_delay=60.0,
            retryable_exceptions=(Exception,),
        )
        def always_fail():
            raise Exception("429 too many requests")

        with patch("personal_assistant.tools.retry_utils.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with pytest.raises(Exception):
                always_fail()

        for delay in sleep_calls:
            assert delay <= 60.0
