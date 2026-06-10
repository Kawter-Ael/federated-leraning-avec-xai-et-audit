"""Client portal - Streamlit application for federated learning local workspace."""
from __future__ import annotations

import streamlit as st

from client_app.auth import current_user, logout, render_auth_screen, require_client_auth, set_state_defaults
from client_app.pages import (
    inject_styles,
    render_dataset,
    render_home,
    render_preprocessing,
    render_results,
    render_training,
    render_xai_audit,
)


def main() -> None:
    st.set_page_config(page_title="Client Local Interface", page_icon=":material/person:", layout="wide")
    inject_styles()
    set_state_defaults()
    if not require_client_auth():
        render_auth_screen()
        return

    user = current_user()
    st.sidebar.markdown("### Session")
    st.sidebar.caption(f"Connected client: `{user['username']}`")
    if st.sidebar.button("Logout", use_container_width=True):
        logout()
        st.rerun()

    navigation_items = ["Home", "Dataset", "Preprocessing", "Training", "XAI / Audit", "Prediction / Results"]
    page = st.sidebar.selectbox("Navigation", navigation_items, index=0)

    if page == "Home":
        render_home()
    elif page == "Dataset":
        render_dataset()
    elif page == "Preprocessing":
        render_preprocessing()
    elif page == "Training":
        render_training()
    elif page == "XAI / Audit":
        render_xai_audit()
    else:
        render_results()


if __name__ == "__main__":
    main()
