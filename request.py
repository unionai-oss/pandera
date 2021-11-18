import requests

r = requests.get("http://127.0.0.1:8000/")
print(r.text)

r = requests.post(
    "http://127.0.0.1:8000/items/",
    json={"name": "Niels", "description": "Hello"},
)
print(r.text)


r = requests.post(
    "http://127.0.0.1:8000/transactions/",
    json={"id": ["1"], "cost": [10.99]},
)
print(r.text)
