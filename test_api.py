import requests
import json

# Test the Flask API
url = "http://localhost:5000/predict"
data = {
    "description": "Find the maximum sum of a subarray using dynamic programming approach",
    "input_desc": "Array of integers with both positive and negative numbers",
    "output_desc": "Single integer representing the maximum sum"
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

# Test with a harder problem
data2 = {
    "description": "Given a graph with weighted edges, find the shortest path between all pairs of vertices using Floyd-Warshall algorithm. Handle negative weights and detect negative cycles.",
    "input_desc": "Adjacency matrix with weights, some may be negative",
    "output_desc": "Matrix of shortest distances between all pairs, or indication of negative cycle"
}

try:
    response2 = requests.post(url, json=data2)
    print(f"\nHard Problem - Status Code: {response2.status_code}")
    print(f"Hard Problem - Response: {response2.json()}")
except Exception as e:
    print(f"Error: {e}")