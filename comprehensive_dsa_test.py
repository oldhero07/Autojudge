import requests
import json
import time

# Comprehensive DSA problem test suite
url = "http://localhost:5000/predict"

# Test problems categorized by difficulty and topic
test_problems = {
    "Easy Array Problems": [
        {
            "description": "Find the maximum element in an array by iterating through all elements once.",
            "input_desc": "Array of integers",
            "output_desc": "Single integer representing the maximum value"
        },
        {
            "description": "Check if an array is sorted in ascending order by comparing adjacent elements.",
            "input_desc": "Array of integers",
            "output_desc": "Boolean value indicating if array is sorted"
        },
        {
            "description": "Count the number of even numbers in an array using a simple loop.",
            "input_desc": "Array of integers",
            "output_desc": "Integer count of even numbers"
        }
    ],
    
    "Medium Array Problems": [
        {
            "description": "Find the maximum sum of a contiguous subarray using Kadane's algorithm.",
            "input_desc": "Array of integers with positive and negative numbers",
            "output_desc": "Integer representing the maximum subarray sum"
        },
        {
            "description": "Rotate an array to the right by k positions using cyclic replacements.",
            "input_desc": "Array of integers and rotation count k",
            "output_desc": "Array rotated k positions to the right"
        },
        {
            "description": "Find two numbers in a sorted array that sum to a target using two pointers.",
            "input_desc": "Sorted array of integers and target sum",
            "output_desc": "Indices of the two numbers that sum to target"
        }
    ],
    
    "Hard Array Problems": [
        {
            "description": "Find the median of two sorted arrays in O(log(min(m,n))) time using binary search.",
            "input_desc": "Two sorted arrays of different sizes",
            "output_desc": "Double representing the median of combined arrays"
        },
        {
            "description": "Find the maximum rectangle area in a histogram using stack-based approach.",
            "input_desc": "Array of integers representing histogram heights",
            "output_desc": "Integer representing maximum rectangle area"
        }
    ],
    
    "Easy String Problems": [
        {
            "description": "Check if a string is a palindrome by comparing characters from both ends.",
            "input_desc": "String of characters",
            "output_desc": "Boolean indicating if string reads same forwards and backwards"
        },
        {
            "description": "Count the frequency of each character in a string using a hash map.",
            "input_desc": "String of characters",
            "output_desc": "Dictionary mapping characters to their frequencies"
        }
    ],
    
    "Medium String Problems": [
        {
            "description": "Find the longest substring without repeating characters using sliding window.",
            "input_desc": "String of characters",
            "output_desc": "Integer length of longest substring without repeats"
        },
        {
            "description": "Check if two strings are anagrams by sorting or character counting.",
            "input_desc": "Two strings to compare",
            "output_desc": "Boolean indicating if strings are anagrams"
        }
    ],
    
    "Hard String Problems": [
        {
            "description": "Find all occurrences of a pattern in text using KMP algorithm with failure function.",
            "input_desc": "Text string and pattern string",
            "output_desc": "List of starting indices where pattern occurs in text"
        },
        {
            "description": "Find the shortest palindrome by adding characters to the front using KMP preprocessing.",
            "input_desc": "String to convert to palindrome",
            "output_desc": "Shortest palindrome string formed by adding characters to front"
        }
    ],
    
    "Easy Tree Problems": [
        {
            "description": "Find the maximum depth of a binary tree using recursive traversal.",
            "input_desc": "Root node of binary tree",
            "output_desc": "Integer representing maximum depth from root to leaf"
        },
        {
            "description": "Check if two binary trees are identical by comparing structure and values.",
            "input_desc": "Root nodes of two binary trees",
            "output_desc": "Boolean indicating if trees are structurally and value-wise identical"
        }
    ],
    
    "Medium Tree Problems": [
        {
            "description": "Validate if a binary tree is a valid binary search tree using inorder traversal.",
            "input_desc": "Root node of binary tree",
            "output_desc": "Boolean indicating if tree satisfies BST property"
        },
        {
            "description": "Find the lowest common ancestor of two nodes in a binary tree using recursive approach.",
            "input_desc": "Root node and two target nodes",
            "output_desc": "Node representing the lowest common ancestor"
        }
    ],
    
    "Hard Tree Problems": [
        {
            "description": "Serialize and deserialize a binary tree using preorder traversal with null markers.",
            "input_desc": "Binary tree root node for serialization, string for deserialization",
            "output_desc": "String representation for serialization, tree root for deserialization"
        },
        {
            "description": "Find the maximum path sum in a binary tree where path can start and end at any nodes.",
            "input_desc": "Root node of binary tree with positive and negative values",
            "output_desc": "Integer representing maximum sum of any path in the tree"
        }
    ],
    
    "Easy Graph Problems": [
        {
            "description": "Perform depth-first search traversal on a graph using recursion or stack.",
            "input_desc": "Graph represented as adjacency list and starting vertex",
            "output_desc": "List of vertices in DFS traversal order"
        },
        {
            "description": "Perform breadth-first search traversal on a graph using queue data structure.",
            "input_desc": "Graph represented as adjacency list and starting vertex",
            "output_desc": "List of vertices in BFS traversal order"
        }
    ],
    
    "Medium Graph Problems": [
        {
            "description": "Detect if a directed graph has a cycle using DFS with recursion stack tracking.",
            "input_desc": "Directed graph represented as adjacency list",
            "output_desc": "Boolean indicating presence of cycle in the graph"
        },
        {
            "description": "Find shortest path in unweighted graph using BFS from source to destination.",
            "input_desc": "Unweighted graph, source vertex, and destination vertex",
            "output_desc": "Integer representing shortest path length or -1 if no path exists"
        }
    ],
    
    "Hard Graph Problems": [
        {
            "description": "Find shortest paths from source to all vertices using Dijkstra's algorithm with priority queue.",
            "input_desc": "Weighted graph with non-negative edges and source vertex",
            "output_desc": "Array of shortest distances from source to all other vertices"
        },
        {
            "description": "Find minimum spanning tree using Kruskal's algorithm with union-find data structure.",
            "input_desc": "Weighted undirected graph represented as edge list",
            "output_desc": "List of edges forming minimum spanning tree and total weight"
        }
    ],
    
    "Easy Dynamic Programming": [
        {
            "description": "Calculate nth Fibonacci number using dynamic programming with memoization.",
            "input_desc": "Non-negative integer n",
            "output_desc": "Integer representing the nth Fibonacci number"
        },
        {
            "description": "Count number of ways to climb stairs with 1 or 2 steps at a time.",
            "input_desc": "Integer representing number of stairs",
            "output_desc": "Integer representing number of distinct ways to reach top"
        }
    ],
    
    "Medium Dynamic Programming": [
        {
            "description": "Solve 0/1 knapsack problem using 2D DP table for optimal item selection.",
            "input_desc": "Item weights, values, and knapsack capacity",
            "output_desc": "Maximum value that can be obtained within weight capacity"
        },
        {
            "description": "Find length of longest common subsequence between two strings using DP.",
            "input_desc": "Two strings to compare",
            "output_desc": "Integer length of longest common subsequence"
        }
    ],
    
    "Hard Dynamic Programming": [
        {
            "description": "Solve edit distance problem using Wagner-Fischer algorithm with 2D DP table.",
            "input_desc": "Two strings to transform between",
            "output_desc": "Minimum number of operations to transform one string to another"
        },
        {
            "description": "Find optimal matrix chain multiplication order using interval DP approach.",
            "input_desc": "Array of matrix dimensions",
            "output_desc": "Minimum number of scalar multiplications needed"
        }
    ],
    
    "Advanced Algorithms": [
        {
            "description": "Implement suffix array construction using SA-IS algorithm in linear time.",
            "input_desc": "String for which to build suffix array",
            "output_desc": "Array of suffix starting positions sorted lexicographically"
        },
        {
            "description": "Solve maximum flow problem using Ford-Fulkerson with Edmonds-Karp implementation.",
            "input_desc": "Flow network with capacities, source and sink vertices",
            "output_desc": "Maximum flow value and the flow assignment on edges"
        },
        {
            "description": "Find strongly connected components using Tarjan's algorithm with DFS and stack.",
            "input_desc": "Directed graph represented as adjacency list",
            "output_desc": "List of strongly connected components as vertex groups"
        }
    ]
}

def test_category(category_name, problems):
    print(f"\n{'='*60}")
    print(f"Testing {category_name}")
    print(f"{'='*60}")
    
    results = []
    for i, problem in enumerate(problems, 1):
        try:
            response = requests.post(url, json=problem)
            if response.status_code == 200:
                result = response.json()
                results.append(result)
                print(f"{i}. Class: {result['class']:<8} Score: {result['score']:>3}/100")
                print(f"   {problem['description'][:70]}...")
            else:
                print(f"{i}. ERROR: HTTP {response.status_code}")
                print(f"   {problem['description'][:70]}...")
        except Exception as e:
            print(f"{i}. ERROR: {e}")
            print(f"   {problem['description'][:70]}...")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    return results

def analyze_results(category_name, results):
    if not results:
        return
    
    classes = [r['class'] for r in results]
    scores = [r['score'] for r in results]
    
    class_counts = {}
    for cls in classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    print(f"\n{category_name} Analysis:")
    print(f"  Average Score: {avg_score:.1f}")
    print(f"  Score Range: {min_score} - {max_score}")
    print(f"  Class Distribution: {class_counts}")

# Run comprehensive tests
print("Starting Comprehensive DSA Problem Testing")
print("Testing model accuracy across different problem types and difficulties")

all_results = {}

for category, problems in test_problems.items():
    results = test_category(category, problems)
    all_results[category] = results
    analyze_results(category, results)

# Overall analysis
print(f"\n{'='*60}")
print("OVERALL ANALYSIS")
print(f"{'='*60}")

total_problems = sum(len(results) for results in all_results.values())
all_scores = []
all_classes = []

for category, results in all_results.items():
    for result in results:
        all_scores.append(result['score'])
        all_classes.append(result['class'])

if all_scores:
    overall_avg = sum(all_scores) / len(all_scores)
    overall_class_counts = {}
    for cls in all_classes:
        overall_class_counts[cls] = overall_class_counts.get(cls, 0) + 1
    
    print(f"Total Problems Tested: {total_problems}")
    print(f"Overall Average Score: {overall_avg:.1f}")
    print(f"Overall Class Distribution: {overall_class_counts}")
    
    # Check if model is working reasonably
    if overall_avg > 0 and len(set(all_classes)) > 1:
        print("\n✅ Model appears to be working - producing varied predictions")
    else:
        print("\n⚠️  Model may need attention - limited prediction variety")

print(f"\n{'='*60}")
print("Test completed!")