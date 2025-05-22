from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI!"}


def test_read_item():
    response = client.get("/items/1")
    assert response.status_code == 200
    assert response.json() == {"item_id": 1, "query": None}

    response = client.get("/items/2?q=testquery")
    assert response.status_code == 200
    assert response.json() == {"item_id": 2, "query": "testquery"}


def test_create_item():
    item_data = {
        "name": "Test Item",
        "description": "A test item",
        "price": 10.5,
        "tax": 1.5,
    }
    response = client.post("/items/", json=item_data)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Item created successfully"
    assert data["item"]["name"] == "Test Item"
    assert data["item"]["description"] == "A test item"
    assert data["item"]["price"] == 10.5
    assert data["item"]["tax"] == 1.5


def test_update_item():
    item_data = {
        "name": "Updated Item",
        "description": "Updated description",
        "price": 20.0,
        "tax": 2.0,
    }
    response = client.put("/items/5", json=item_data)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Item updated"
    assert data["item_id"] == 5
    assert data["item"]["name"] == "Updated Item"


def test_delete_item_success():
    response = client.delete("/items/3")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Item deleted successfully"
    assert data["item_id"] == 3


def test_delete_item_invalid_id():
    response = client.delete("/items/0")
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid item ID"