import sys
import os

# Ensure the project root is on sys.path so 'from src.xxx import ...' works
# regardless of how pytest is invoked (e.g., from CI, from project root, etc.)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
