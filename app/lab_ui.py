"""Small UI helpers shared by the lab's pages (Streamlit only; computation lives in the library).

SPEC §8: the app trains small models live and loads everything else from results/ through
runs.py; when an artifact is missing it shows the exact CLI command that creates it.
"""

from __future__ import annotations

import streamlit as st

PAPER_URL = "https://transformer-circuits.pub/2022/toy_model/index.html"


def page_setup(title: str) -> None:
    """Common page config and sidebar note."""
    st.set_page_config(page_title=f"{title} · superposition lab", layout="wide")
    with st.sidebar:
        st.caption(
            "An independent replication of "
            f"[*Toy Models of Superposition*]({PAPER_URL}) (Elhage et al., 2022). "
            "Page numbers refer to the 62-page PDF print of the article."
        )


def missing_artifact(what: str, command: str) -> None:
    """Tell the user what is missing and the exact command that generates it."""
    st.info(f"{what} not found. Generate it with:")
    st.code(command, language="bash")
