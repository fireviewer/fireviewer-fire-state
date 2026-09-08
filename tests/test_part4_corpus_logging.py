from __future__ import annotations

import logging
from pathlib import Path

import pytest

from fireviewer_fire_state.part4_calibration_campaign import bounded_scratch


def test_scratch_redacts_signed_hub_retry_urls_and_restores_filters(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("huggingface_hub.utils._http")
    previous = list(logger.filters)
    with bounded_scratch(parent=tmp_path), caplog.at_level(logging.WARNING):
        logger.warning(
            "HTTP Error %s while requesting PUT %s",
            503,
            "https://example.com/object?X-Amz-Signature=private-test-value&key=secret",
        )
    assert "private-test-value" not in caplog.text
    assert "key=secret" not in caplog.text
    assert "HTTP Error 503" in caplog.text
    assert "https://example.com/object?[redacted]" in caplog.text
    assert logger.filters == previous
    assert not list(tmp_path.iterdir())
