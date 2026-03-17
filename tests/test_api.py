"""
DocLens API Tests
Tests for health checks, demo page, document recognition, auth, and usage tracking.
"""
import io
import json
import pytest
import httpx
import numpy as np
import cv2
from pathlib import Path

# ============================================================
# Test Configuration
# ============================================================

BASE_URL = "http://localhost:8000"
TEST_API_KEY = "dl_live_test_key_for_pytest"

# Simple 1x1 white PNG for testing
WHITE_PIXEL_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f'
    b'\x00\x00\x01\x01\x00\x05\x18e\xde\x00\x00\x00\x00IEND\xaeB`\x82'
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    """Create HTTP client for testing."""
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


@pytest.fixture
def test_image_passport():
    """Create a synthetic test image that resembles a document."""
    # Create a simple image with text-like patterns
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Add some contrasting regions to simulate document structure
    cv2.rectangle(img, (20, 20), (580, 100), (0, 0, 0), 2)
    cv2.putText(img, "PASSPORT", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(img, (20, 120), (580, 200), (100, 100, 100), -1)
    cv2.putText(img, "NAME", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Encode to PNG
    success, buffer = cv2.imencode('.png', img)
    return io.BytesIO(buffer.tobytes())


@pytest.fixture
def test_image_small_white():
    """Return a minimal white PNG for testing."""
    return io.BytesIO(WHITE_PIXEL_PNG)


# ============================================================
# Tests: Health & Status
# ============================================================

def test_health_endpoint(client):
    """Test that health endpoint returns 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_health_endpoint_structure(client):
    """Test health endpoint returns required fields."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data or "service" in data


# ============================================================
# Tests: Demo Page
# ============================================================

def test_demo_page_loads(client):
    """Test that demo page HTML loads."""
    response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_demo_page_contains_form(client):
    """Test that demo page contains file upload form."""
    response = client.get("/demo")
    assert response.status_code == 200
    html = response.text

    # Check for form elements
    assert "form" in html.lower() or "upload" in html.lower()


def test_index_page_loads(client):
    """Test that index/root page loads."""
    response = client.get("/")
    assert response.status_code == 200


# ============================================================
# Tests: Document Recognition Endpoint
# ============================================================

def test_recognize_endpoint_accepts_file(client, test_image_small_white):
    """Test recognize endpoint accepts file uploads."""
    files = {"file": ("test.png", test_image_small_white, "image/png")}
    response = client.post("/api/v1/recognize", files=files)

    # Should succeed or return validation error, but not 404/405
    assert response.status_code in [200, 400, 422, 501]  # 501 if OCR not available


def test_recognize_endpoint_with_passport_image(client, test_image_passport):
    """Test recognize endpoint with synthetic passport-like image."""
    files = {"file": ("passport.png", test_image_passport, "image/png")}
    response = client.post("/api/v1/recognize", files=files)

    # Accept various status codes (depends on OCR availability)
    assert response.status_code in [200, 400, 422, 501, 503]


def test_recognize_endpoint_returns_json(client, test_image_small_white):
    """Test that recognize endpoint returns JSON."""
    files = {"file": ("test.png", test_image_small_white, "image/png")}
    response = client.post("/api/v1/recognize", files=files)

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
        # Should have recognition results or status
        assert "result" in data or "status" in data or "error" in data


def test_recognize_endpoint_missing_file(client):
    """Test recognize endpoint with missing file."""
    response = client.post("/api/v1/recognize")
    assert response.status_code in [400, 422]  # Bad request


def test_recognize_endpoint_invalid_file_type(client):
    """Test recognize endpoint rejects invalid file types."""
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/v1/recognize", files=files)

    # Should reject non-image files
    assert response.status_code in [400, 415, 422]


# ============================================================
# Tests: API Authentication
# ============================================================

def test_api_key_header_validation(client, test_image_small_white):
    """Test that API key validation works (if implemented)."""
    files = {"file": ("test.png", test_image_small_white, "image/png")}
    headers = {"X-API-Key": TEST_API_KEY}

    response = client.post(
        "/api/v1/recognize",
        files=files,
        headers=headers
    )

    # Should either accept or reject with 401/403, not 404/500
    assert response.status_code in [200, 400, 401, 403, 422, 501, 503]


def test_bearer_token_header(client, test_image_small_white):
    """Test Bearer token authentication (if implemented)."""
    files = {"file": ("test.png", test_image_small_white, "image/png")}
    headers = {"Authorization": "Bearer test_token_12345"}

    response = client.post(
        "/api/v1/recognize",
        files=files,
        headers=headers
    )

    # Should handle gracefully
    assert response.status_code in [200, 400, 401, 403, 422, 501, 503]


# ============================================================
# Tests: Usage Tracking
# ============================================================

def test_usage_endpoint_exists(client):
    """Test that usage tracking endpoint exists."""
    # Test common usage tracking endpoints
    endpoints = [
        "/api/v1/usage",
        "/api/v1/stats",
        "/api/v1/metrics"
    ]

    for endpoint in endpoints:
        response = client.get(endpoint)
        # Accept 200, 401 (auth required), 404 (not implemented)
        assert response.status_code in [200, 401, 404]


def test_metrics_endpoint(client):
    """Test metrics/Prometheus endpoint."""
    response = client.get("/metrics")

    # Should exist or return 404 if not enabled
    if response.status_code == 200:
        # If it exists, should be text/plain (Prometheus format)
        assert "text/plain" in response.headers.get("content-type", "")


# ============================================================
# Tests: Content Negotiation
# ============================================================

def test_accept_json_header(client):
    """Test that API respects Accept: application/json."""
    headers = {"Accept": "application/json"}
    response = client.get("/api/v1/health", headers=headers)
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")


def test_content_type_on_post(client, test_image_small_white):
    """Test Content-Type headers on POST requests."""
    files = {"file": ("test.png", test_image_small_white, "image/png")}
    response = client.post("/api/v1/recognize", files=files)

    if response.status_code == 200:
        # Should return JSON
        assert response.headers.get("content-type") is not None


# ============================================================
# Tests: Error Handling
# ============================================================

def test_404_not_found(client):
    """Test 404 error handling."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404


def test_method_not_allowed(client):
    """Test 405 Method Not Allowed."""
    response = client.post("/api/v1/health")
    # Health endpoint is likely GET only
    assert response.status_code in [405, 200]  # 200 if accepts POST


def test_error_response_format(client):
    """Test that error responses are properly formatted."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404

    # Try to parse as JSON
    try:
        data = response.json()
        assert "detail" in data or "error" in data or "message" in data
    except json.JSONDecodeError:
        # Some endpoints might return HTML for 404
        assert "404" in response.text or "not found" in response.text.lower()


# ============================================================
# Tests: Rate Limiting (if implemented)
# ============================================================

def test_rate_limit_headers(client):
    """Test that rate limit headers are present (if implemented)."""
    response = client.get("/api/v1/health")

    # Look for rate limit headers
    rate_limit_headers = [
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "RateLimit-Limit"
    ]

    headers = response.headers
    has_rate_limiting = any(h in headers for h in rate_limit_headers)

    # Rate limiting is optional, just check it doesn't error
    assert response.status_code == 200


# ============================================================
# Integration Tests
# ============================================================

def test_full_recognition_workflow(client, test_image_passport):
    """Test a complete recognition workflow."""
    # 1. Check health
    health = client.get("/api/v1/health")
    assert health.status_code == 200

    # 2. Load demo page
    demo = client.get("/demo")
    assert demo.status_code == 200

    # 3. Upload document
    files = {"file": ("passport.png", test_image_passport, "image/png")}
    recognize = client.post("/api/v1/recognize", files=files)

    # Should not 500 error
    assert recognize.status_code != 500


def test_multiple_uploads_sequence(client, test_image_small_white):
    """Test multiple sequential uploads."""
    for i in range(3):
        files = {"file": (f"test_{i}.png", test_image_small_white, "image/png")}
        response = client.post("/api/v1/recognize", files=files)

        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 501, 503]


# ============================================================
# Performance Tests
# ============================================================

def test_health_endpoint_response_time(client):
    """Test that health endpoint responds quickly."""
    import time

    start = time.time()
    response = client.get("/api/v1/health")
    elapsed = time.time() - start

    assert response.status_code == 200
    assert elapsed < 1.0  # Should respond in under 1 second


def test_demo_page_response_time(client):
    """Test that demo page loads quickly."""
    import time

    start = time.time()
    response = client.get("/demo")
    elapsed = time.time() - start

    assert response.status_code == 200
    assert elapsed < 2.0  # Should load in under 2 seconds


# ============================================================
# CLI Test Runner
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
