"""DataNova core package.

Everything in ``core`` is plain Python (pandas / numpy / scikit-learn / plotly)
and does not import Streamlit, so it can be unit-tested and reused outside the
web UI. The Streamlit pages live in ``views/``.

Designed & developed by Gitesh Kund.
"""

__version__ = "1.0.0"
APP_NAME = "DataNova"
APP_TAGLINE = "Smart Analytics & Auto-Dashboard Studio"
AUTHOR = "Gitesh Kund"
CREDIT = f"Designed & developed by {AUTHOR}"
