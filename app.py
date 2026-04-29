"""
Vercel entrypoint for the Workflow Generator FastAPI application.
This file is required by Vercel's FastAPI framework detection.
"""

from src.api.server import app

__all__ = ["app"]
