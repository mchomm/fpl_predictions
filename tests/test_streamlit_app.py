"""Browser-script smoke tests for the deployed Streamlit interface."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.skipif(
    not Path("deployment/current/manifest.json").is_file(),
    reason="serving bundle is not available",
)
def test_streamlit_app_loads_and_runs_rating_and_optimizer() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "FPL Squad Lab"

    next(
        item for item in app.button if item.label == "Generate best possible squad"
    ).click().run()
    assert not app.exception
    assert any(item.label == "Formation" for item in app.metric)

    next(item for item in app.button if item.label == "Rate this squad").click().run()
    assert not app.exception
    assert any(item.label == "Overall" for item in app.metric)

    next(item for item in app.button if item.label == "Suggest transfers").click().run()
    assert not app.exception
    assert {item.label for item in app.download_button} == {
        "Download rating JSON",
        "Download optimization JSON",
    }
