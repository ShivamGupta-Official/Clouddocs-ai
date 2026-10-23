"""
Tests for the health check endpoint.

Verifies that:
- The /health endpoint returns 200
- The response contains expected fields
- No sensitive information is leaked
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_200():
    """The health endpoint should always return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_has_required_fields():
    """The health response must include status, application name, and components."""
    response = client.get("/health")
    data = response.json()

    assert data["status"] == "healthy"
    assert data["application"] == "CloudDocs AI"
    assert "version" in data
    assert "components" in data
    assert "configuration" in data


def test_health_response_has_component_status():
    """Each component should report its status."""
    response = client.get("/health")
    components = response.json()["components"]

    assert "api" in components
    assert "metadata_db" in components
    assert "vector_store" in components
    assert "llm" in components


def test_health_does_not_expose_secrets():
    """The health response must never contain API keys or passwords."""
    response = client.get("/health")
    response_text = response.text.lower()

    # These strings should never appear in the health response
    sensitive_patterns = ["api_key", "password", "secret", "token", "credential"]
    for pattern in sensitive_patterns:
        assert pattern not in response_text, (
            f"Health endpoint may be leaking sensitive data: found '{pattern}'"
        )


def test_health_configuration_shows_safe_info():
    """Configuration section should show model names but not keys."""
    response = client.get("/health")
    config = response.json()["configuration"]

    assert "embedding_model" in config
    assert "llm_provider" in config
    assert "chunk_size" in config
    assert "max_upload_size_mb" in config
