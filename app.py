# Root-level entrypoint for Streamlit Cloud
# Redirects to the actual app in streamlit_app/app.py

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Run the actual app
from streamlit_app.app import main
main()