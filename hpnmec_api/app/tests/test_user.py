import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def user_data():
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "TestPass123",
        "full_name": "Test User"
    }

def test_register_success(user_data):
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 201 or response.status_code == 200
    assert response.json()["status"] == "success" or response.json().get("user")

def test_register_duplicate(user_data):
    client.post("/api/v1/auth/register", json=user_data)
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 400 or response.json()["status"] == "error"

def test_login_success(user_data):
    client.post("/api/v1/auth/register", json=user_data)
    response = client.post("/api/v1/auth/login", json={"email": user_data["email"], "password": user_data["password"]})
    assert response.status_code == 200
    assert "token" in response.json() or response.json()["status"] == "success"

def test_login_wrong_password(user_data):
    client.post("/api/v1/auth/register", json=user_data)
    response = client.post("/api/v1/auth/login", json={"email": user_data["email"], "password": "WrongPass123"})
    assert response.status_code == 401 or response.json()["status"] == "error"
