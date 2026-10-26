"""Automated smoke tests for Streamlit web application using AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "app" / "streamlit_app.py")


def test_streamlit_app_loads_and_runs_default_demo():
    """Verify that Streamlit app initializes, runs bundled demo, and renders without exceptions."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=40)

    # Assert no unhandled runtime exceptions
    assert len(at.exception) == 0, f"Streamlit app raised exceptions: {[e.value for e in at.exception]}"

    # Verify key interactive elements are present
    assert len(at.dataframe) >= 1, "Expected at least 1 dataframe (matches table) to render"
    assert len(at.metric) >= 4, "Expected summary metrics (Source, Target, Matches, etc.)"

    # Verify matches table content
    matches_df = at.dataframe[0].value
    assert "Source Column" in matches_df.columns
    assert "Target Column" in matches_df.columns
    assert "Confidence" in matches_df.columns
    assert len(matches_df) > 0


def test_streamlit_app_switch_demo_pair():
    """Verify selecting a different bundled demo pair updates and runs cleanly."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=40)
    assert len(at.exception) == 0

    # Switch selectbox to schema_pair_2
    if at.sidebar.selectbox:
        demo_select = at.sidebar.selectbox[0]
        demo_select.select("schema_pair_2").run(timeout=40)

        assert len(at.exception) == 0
        assert len(at.dataframe) >= 1
