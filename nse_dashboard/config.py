"""
Angel One credentials — read from Streamlit secrets first, then env vars.

In Streamlit Cloud: add the [angel_one] section to your app's Secrets in the
dashboard (Settings → Secrets). Format:

    [angel_one]
    api_key     = "YOUR_API_KEY"
    secret_key  = "YOUR_SECRET_KEY"
    client_id   = "YOUR_CLIENT_ID"
    password    = "YOUR_PASSWORD"
    mpin        = "YOUR_MPIN"
    totp_secret = "YOUR_TOTP_SECRET"

For local dev: set env vars AO_API_KEY, AO_SECRET_KEY, AO_CLIENT_ID,
AO_PASSWORD, AO_MPIN, AO_TOTP_SECRET.
"""

import os


def _get_angel_one():
    try:
        import streamlit as st
        s = st.secrets.get("angel_one", {})
        if s.get("api_key"):
            return {k: str(s[k]) for k in
                    ("api_key", "secret_key", "client_id", "password", "mpin", "totp_secret")}
    except Exception:
        pass
    return {
        "api_key":     os.getenv("AO_API_KEY", ""),
        "secret_key":  os.getenv("AO_SECRET_KEY", ""),
        "client_id":   os.getenv("AO_CLIENT_ID", ""),
        "password":    os.getenv("AO_PASSWORD", ""),
        "mpin":        os.getenv("AO_MPIN", ""),
        "totp_secret": os.getenv("AO_TOTP_SECRET", ""),
    }


ANGEL_ONE = _get_angel_one()
