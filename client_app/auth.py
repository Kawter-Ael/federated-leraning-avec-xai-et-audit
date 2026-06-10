"""Authentication and session management for the client portal."""

from __future__ import annotations

import streamlit as st

from shared.auth_store import authenticate_client_user


def set_state_defaults() -> None:
    for key in (
        "authenticated",
        "auth_user",
        "auth_message",
        "client_runtime_config",
        "client_validation",
        "client_phase2",
        "client_training",
        "client_xai",
        "client_audit",
    ):
        st.session_state.setdefault(key, None)
    if st.session_state["authenticated"] is None:
        st.session_state["authenticated"] = False


def current_user() -> dict[str, str] | None:
    user = st.session_state.get("auth_user")
    return user if st.session_state.get("authenticated") and user else None


def require_client_auth() -> bool:
    return bool(
        st.session_state.get("authenticated") and st.session_state.get("auth_user")
    )


def logout() -> None:
    st.session_state["authenticated"] = False
    st.session_state["auth_user"] = None
    st.session_state["auth_message"] = None
    for key in (
        "client_runtime_config",
        "client_validation",
        "client_phase2",
        "client_training",
        "client_xai",
        "client_audit",
    ):
        st.session_state[key] = None


def render_auth_screen() -> None:
    st.markdown(
        """
        <style>
        /* Hide Streamlit chrome */
        #MainMenu, footer, header,
        [data-testid="stToolbar"],
        [data-testid="stDeployButton"],
        .stDeployButton { display: none !important; }

        /* Page background */
        .stApp {
            background: linear-gradient(140deg, #e8eef5 0%, #dde6ef 100%);
        }

        /* Vertically center card */
        section[data-testid="stMain"] {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            min-height: 100vh !important;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
        }

        /* ── The single login card ── */
        /* Style the Streamlit form container as the card so widgets stay inside */
        [data-testid="stForm"] {
            background: #ffffff !important;
            border-radius: 20px !important;
            box-shadow: 0 20px 56px rgba(29,43,58,0.13), 0 4px 16px rgba(29,43,58,0.07) !important;
            border: 1px solid rgba(31,78,121,0.09) !important;
            padding: 2.25rem 2rem 1.75rem 2rem !important;
        }

        /* Badge — force white via all specificity paths */
        .lc-brand,
        [data-testid="stMarkdownContainer"] .lc-brand,
        [data-testid="stForm"] .lc-brand {
            display: inline-block;
            letter-spacing: .13em;
            text-transform: uppercase;
            font-size: .68rem;
            font-weight: 700;
            color: #ffffff !important;
            background: linear-gradient(135deg, #1f4e79 0%, #1b7a73 100%);
            padding: .22rem .75rem;
            border-radius: 999px;
            margin-bottom: .75rem;
        }
        .lc-brand *,
        [data-testid="stMarkdownContainer"] .lc-brand *,
        [data-testid="stForm"] .lc-brand * {
            color: #ffffff !important;
        }
        .lc-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #1d2b3a;
            margin: 0 0 .2rem 0;
            line-height: 1.2;
        }
        .lc-sub {
            color: #5b697a;
            font-size: .88rem;
            margin-bottom: 1.4rem;
            line-height: 1.55;
        }
        .lc-divider {
            border: none;
            border-top: 1px solid #eef1f5;
            margin: 0 0 1.2rem 0;
        }
        .lc-note {
            color: #8a9bac;
            font-size: .78rem;
            text-align: center;
            margin-top: .5rem;
        }

        /* Login button — gradient, white text (target button + all inner elements) */
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stForm"] button[kind="primaryFormSubmit"],
        [data-testid="stForm"] [data-testid="baseButton-primaryFormSubmit"] {
            background: linear-gradient(135deg, #1f4e79 0%, #1b7a73 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: .97rem !important;
            height: 2.65rem !important;
            letter-spacing: .02em !important;
        }
        [data-testid="stFormSubmitButton"] button p,
        [data-testid="stFormSubmitButton"] button span,
        [data-testid="stFormSubmitButton"] button div,
        [data-testid="stForm"] button[kind="primaryFormSubmit"] p,
        [data-testid="stForm"] button[kind="primaryFormSubmit"] span,
        [data-testid="stForm"] button[kind="primaryFormSubmit"] div {
            color: #ffffff !important;
        }
        [data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover p,
        [data-testid="stFormSubmitButton"] button:hover span,
        [data-testid="stFormSubmitButton"] button:hover div,
        [data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
        [data-testid="stForm"] [data-testid="baseButton-primaryFormSubmit"]:hover {
            opacity: .88 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _left, center, _right = st.columns([1, 1.05, 1])
    with center:
        # Everything inside st.form → single card in DOM
        with st.form("login-form", clear_on_submit=False):
            st.markdown(
                """
                <span class="lc-brand" style="color:#ffffff !important;">ENSAJ FL</span>
                <div class="lc-title">Client sign in</div>
                <div class="lc-sub">Access the federated learning workspace with your provisioned academic client account.</div>
                <hr class="lc-divider">
                """,
                unsafe_allow_html=True,
            )
            username = st.text_input(
                "Username",
                key="login-username",
                placeholder="client_1",
            )
            password = st.text_input(
                "Password",
                type="password",
                key="login-password",
                placeholder="Enter your password",
            )
            submitted = st.form_submit_button("Log in", use_container_width=True)
            st.markdown(
                '<div class="lc-note">Only provisioned client accounts can access this workspace.</div>',
                unsafe_allow_html=True,
            )

        auth_message = st.session_state.get("auth_message")
        if auth_message:
            level, message = auth_message
            if level == "success":
                st.success(message)
            else:
                st.error(message)

    if submitted:
        try:
            ok, user, message = authenticate_client_user(username, password)
        except Exception as exc:
            ok, user, message = False, None, f"Auth connection failed: {exc}"
        st.session_state["auth_message"] = ("success" if ok else "error", message)
        if ok:
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = user
            st.rerun()
        else:
            st.rerun()
