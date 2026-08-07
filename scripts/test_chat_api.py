"""Comprehensive test script to test all API endpoints and multi-turn chat functionality.

Modes supported:
1. Direct FastAPI TestClient (no server needed)
2. HTTP requests to running server (e.g. http://localhost:8000)

Usage:
  - Run offline with TestClient:
      uv run python scripts/test_chat_api.py
  - Run against active server:
      uv run python scripts/test_chat_api.py --live http://localhost:8000
"""

import sys
import json
import argparse
from typing import Any


def run_tests_with_client(client: Any) -> None:
    print("=" * 60)
    print("🚀 Starting Polo API Full Functionality Test")
    print("=" * 60)

    # 1. Test Root Endpoint GET /
    print("\n1. Testing GET / (Root Endpoint)...")
    res = client.get("/")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200, "Root endpoint failed"

    # 2. Test Health Endpoint GET /health
    print("\n2. Testing GET /health...")
    res = client.get("/health")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200, "Health check failed"

    # 3. Test Refresh Catalog POST /admin/refresh-catalog
    print("\n3. Testing POST /admin/refresh-catalog...")
    res = client.post("/admin/refresh-catalog")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200, "Refresh catalog failed"

    # 4. Test Chat - Step 1: Initial greeting & information
    print("\n4. Testing POST /chat (Turn 1 - Initial message)...")
    payload1 = {
        "message": "Hi, I have a 3-year-old Golden Retriever named Max who weighs 65 lbs."
    }
    res = client.post("/chat", json=payload1)
    data1 = res.json()
    print(f"Status Code: {res.status_code}")
    print(f"Full Response:\n{json.dumps(data1, indent=2)}")

    assert res.status_code == 200, "Chat request 1 failed"
    assert data1.get("status") is True, "Chat response status was not True"
    session_id = data1.get("session_id")
    assert session_id, "No session_id returned in Chat response"
    print(f"✅ Generated Session ID: {session_id}")

    # 5. Test Chat - Step 2: Multi-turn memory check using session_id
    print("\n5. Testing POST /chat (Turn 2 - Context memory check)...")
    payload2 = {
        "session_id": session_id,
        "message": "What food recommendations do you have for Max?"
    }
    res = client.post("/chat", json=payload2)
    data2 = res.json()
    print(f"Status Code: {res.status_code}")
    print(f"Full Response:\n{json.dumps(data2, indent=2)}")

    assert res.status_code == 200, "Chat request 2 failed"
    assert data2.get("session_id") == session_id, "Session ID mismatch in turn 2"

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Polo Assistant API")
    parser.add_argument(
        "--live",
        type=str,
        default=None,
        help="Base URL of live server (e.g. http://localhost:8000). If omitted, uses FastAPI TestClient directly.",
    )
    args = parser.parse_args()

    if args.live:
        import requests

        class LiveClient:
            def __init__(self, base_url: str):
                self.base_url = base_url.rstrip("/")

            def get(self, path: str):
                r = requests.get(f"{self.base_url}{path}")
                return r

            def post(self, path: str, json: dict = None):
                r = requests.post(f"{self.base_url}{path}", json=json)
                return r

        print(f"Testing against live server at {args.live}...")
        client = LiveClient(args.live)
        run_tests_with_client(client)
    else:
        from fastapi.testclient import TestClient
        from main import app

        print("Testing in-memory with FastAPI TestClient...")
        client = TestClient(app)
        run_tests_with_client(client)


if __name__ == "__main__":
    main()
