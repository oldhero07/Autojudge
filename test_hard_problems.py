import requests
import json

# Test with clearly hard algorithmic problems
url = "http://localhost:5000/predict"

hard_problems = [
    {
        "description": "Given a tree with weighted edges, find the minimum vertex cover using dynamic programming on trees. The vertex cover must include vertices such that every edge has at least one endpoint in the cover.",
        "input_desc": "Tree structure with n vertices, weighted edges, and adjacency list representation",
        "output_desc": "Minimum weight vertex cover and the vertices included in the solution"
    },
    {
        "description": "Implement the Hungarian algorithm to solve the assignment problem in O(n^3) time. Given a cost matrix, find the minimum cost perfect matching in a bipartite graph.",
        "input_desc": "n x n cost matrix where entry (i,j) represents cost of assigning worker i to job j",
        "output_desc": "Minimum total cost and the optimal assignment of workers to jobs"
    },
    {
        "description": "Solve the traveling salesman problem using dynamic programming with bitmasks. Find the shortest Hamiltonian cycle that visits all cities exactly once.",
        "input_desc": "Distance matrix between n cities (n <= 20)",
        "output_desc": "Minimum tour length and the sequence of cities in the optimal tour"
    }
]

for i, problem in enumerate(hard_problems, 1):
    try:
        response = requests.post(url, json=problem)
        result = response.json()
        print(f"Hard Problem {i}: {result['class']} - {result['score']}/100")
        print(f"  Description: {problem['description'][:80]}...")
        print()
    except Exception as e:
        print(f"Error with problem {i}: {e}")