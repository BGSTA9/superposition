"""Small UI helpers shared by the lab's pages (Streamlit only; computation lives in the library).

SPEC §8: the app trains small models live and loads everything else from results/ through
runs.py; when an artifact is missing it shows the exact CLI command that creates it.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

PAPER_URL = "https://transformer-circuits.pub/2022/toy_model/index.html"

LAB_NAME = "Argo Navis Research Lab"
LOGO_PATH = Path(__file__).resolve().parents[1] / "ANRL.svg"

# Pins the credit to the sidebar's lower-left corner on every page: the sidebar's content becomes
# a full-height flex column, and the credit's element is ordered last with `margin-top: auto`, so
# it sits at the bottom when the sidebar is short and after the content when it is long. It
# follows the sidebar's width when the sidebar is resized or collapsed.
_FOOTER_CSS = """
<style>
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] {
  flex: 1 0 auto; display: flex; flex-direction: column; padding-bottom: 1.25rem;
}
[data-testid="stSidebarUserContent"] > div { flex: 1 0 auto; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { flex: 1 0 auto; }
[data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]:has(.lab-credit) {
  order: 999; margin-top: auto;
}
.lab-credit { display: flex; align-items: center; gap: 0.6rem; }
.lab-credit img { height: 2.25rem; width: auto; flex: none; }
.lab-credit span { font-size: 0.85rem; font-weight: 600; line-height: 1.25; }
</style>
"""


@lru_cache(maxsize=1)
def _logo_data_uri() -> str | None:
    """The lab's logo (ANRL.svg at the repository root) as a data URI, or None if it is missing."""
    try:
        return "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except OSError:
        return None


def lab_credit() -> None:
    """The lab's logo, then its name, in the sidebar's lower-left corner."""
    logo = _logo_data_uri()
    image = f'<img src="{logo}" alt="" aria-hidden="true">' if logo else ""
    st.markdown(
        f'{_FOOTER_CSS}<div class="lab-credit">{image}<span>{LAB_NAME}</span></div>',
        unsafe_allow_html=True,
    )


def page_setup(title: str) -> None:
    """Common page config, sidebar note and the lab's credit."""
    st.set_page_config(page_title=f"{title} · superposition lab", layout="wide")
    with st.sidebar:
        st.caption(
            "An independent replication of "
            f"[*Toy Models of Superposition*]({PAPER_URL}) (Elhage et al., 2022). "
            "Page numbers refer to the 62-page PDF print of the article."
        )
        lab_credit()


def missing_artifact(what: str, command: str) -> None:
    """Tell the user what is missing and the exact command that generates it."""
    st.info(f"{what} not found. Generate it with:")
    st.code(command, language="bash")
