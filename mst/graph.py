import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

        # Validate adjacency matrix
        self._validate_adjacency_matrix()

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def _validate_adjacency_matrix(self):
        """
        Ensures that the adjacency matrix is valid:
        - It must be a square 2D numpy array.
        - It must represent an undirected graph (symmetric matrix).
        - The graph must be connected.
        """
        mat = self.adj_mat

        # Check if the adjacency matrix is a numpy array
        if not isinstance(mat, np.ndarray):
            raise TypeError('Adjacency matrix must be a numpy array!')

        # Check if the matrix is 2D and square
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError('Adjacency matrix must be a square 2D numpy array!')

        # Check for symmetry (undirected graph)
        if not np.allclose(mat, mat.T):
            raise ValueError('Adjacency matrix must be symmetric!')

        # Check for connectivity using the is_connected method
        if not self.is_connected():
            raise ValueError('Graph is not connected!')

    def is_connected(self):
        """
        Check if the graph represented by the adjacency matrix is connected.
        Uses DFS (Depth-First Search) to check connectivity.
        """
        num_vertices = self.adj_mat.shape[0]
        visited = set()

        def dfs(node):
            visited.add(node)
            for neighbor in range(num_vertices):
                if self.adj_mat[node, neighbor] != 0 and neighbor not in visited:
                    dfs(neighbor)

        dfs(0)  # Start traversal from the first node
        return len(visited) == num_vertices

    def construct_mst(self):
        """
        Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        # Construct the MST using Prim 's algorithm and store it as an adjacency matrix in self.mst
        num_vertices = self.adj_mat.shape[0]

        # Edge case: single-node graph
        if num_vertices == 1:
            self.mst = np.zeros((1, 1))
            return

        # Initialize MST and visited nodes
        self.mst = np.zeros((num_vertices, num_vertices))
        visited = set()
        min_heap = []

        # To track processed edges
        processed_edges = np.zeros((num_vertices, num_vertices), dtype=bool)

        # Start from vertex 0
        start_vertex = 0
        visited.add(start_vertex)

        # Add all edges from the start vertex to the heap
        for neighbor in range(num_vertices):
            if start_vertex != neighbor and self.adj_mat[start_vertex, neighbor] != 0:
                heapq.heappush(min_heap, (self.adj_mat[start_vertex, neighbor], start_vertex, neighbor))
                processed_edges[start_vertex, neighbor] = True
                processed_edges[neighbor, start_vertex] = True

        while min_heap:
            # Get the smallest edge from the heap
            weight, u, v = heapq.heappop(min_heap)

            # Skip if the destination vertex is already visited
            if v in visited:
                continue

            # Mark the vertex as visited and add the edge to the MST
            visited.add(v)
            self.mst[u, v] = weight
            self.mst[v, u] = weight

            # Add all edges from the newly added vertex to the heap
            for neighbor in range(num_vertices):
                if (
                        neighbor not in visited and
                        self.adj_mat[v, neighbor] != 0 and
                        not processed_edges[v, neighbor]
                ):
                    heapq.heappush(min_heap, (self.adj_mat[v, neighbor], v, neighbor))
                    processed_edges[v, neighbor] = True
                    processed_edges[neighbor, v] = True