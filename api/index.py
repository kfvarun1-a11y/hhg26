"""
Vercel Serverless Entrypoint for Voice-Enabled RAG FastAPI Backend.
Exposes the FastAPI `app` instance for Vercel Python runtime (@vercel/python).
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
