"""Ensure manifold_tools is importable from tests/ without installation."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
