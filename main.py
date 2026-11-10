"""
main.py
Root entrypoint for the Retail Product Demand Estimation and Stockout Risk Analysis application.
Can be executed directly via:
    python main.py
Or via Uvicorn:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

import os
import sys
import uvicorn

# Ensure the project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app

if __name__ == "__main__":
    print("=" * 70)
    print("STARTING RETAIL DEMAND ESTIMATION & STOCKOUT RISK SERVER")
    print("Local Web App URL:      http://127.0.0.1:8000")
    print("Interactive API Docs:   http://127.0.0.1:8000/docs")
    print("ReDoc Documentation:    http://127.0.0.1:8000/redoc")
    print("=" * 70)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
