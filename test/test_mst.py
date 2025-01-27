import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances

def check_mst(adj_mat: np.ndarray,
              mst: np.ndarray,
              expected_weight: int,
              allowed_error: float = 0.0001):
    """
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot
    simply check for equality against a known MST of a graph.

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    Added additional assertions to ensure the correctness of your MST implementation.
    (For example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?)
    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    # Checking total and expected weights
    total = 0
    for i in range(mst.shape[0]):
        for j in range(i + 1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # Handle single-node graph
    num_nodes = adj_mat.shape[0]
    if num_nodes == 1:
        assert mst.shape == (1, 1) and mst[0, 0] == 0, 'MST for a single-node graph should be a 1x1 zero matrix'
        return

    # Check symmetry
    assert np.allclose(mst, mst.T), 'Proposed MST adjacency matrix is not symmetric'

    # Check number of edges in MST
    num_edges = np.count_nonzero(mst) / 2
    assert num_edges == num_nodes - 1, 'MST does not have the correct number of edges'

    # Validate MST weight is less than or equal to original graph weight
    assert np.sum(mst) <= np.sum(adj_mat), 'Proposed MST weight exceeds graph weight'

    # Check connectivity of MST using Graph's is_connected method
    graph = Graph(mst)
    assert graph.is_connected(), 'Proposed MST is not connected'

def test_mst_small():
    """
    Unit test for the construction of a minimum spanning tree on a small graph.
    """
    file_path = 'data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)

def test_mst_single_cell_data():
    """
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.
    https://bioconductor.org/packages/release/bioc/html/slingshot.html
    """
    file_path = 'data/slingshot_example.txt'
    coords = np.loadtxt(file_path)  # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords)  # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)

def test_mst_student():
    """
    Wrote at least one unit test for MST construction
    """
    # Test with a single-node graph
    single_node_graph = np.zeros((1, 1))
    single_node_g = Graph(single_node_graph)
    single_node_g.construct_mst()
    check_mst(single_node_g.adj_mat, single_node_g.mst, expected_weight=0)

    # Test with a small valid graph
    small_graph = np.array([[0, 2, 0, 6, 0],
                            [2, 0, 3, 8, 5],
                            [0, 3, 0, 0, 7],
                            [6, 8, 0, 0, 9],
                            [0, 5, 7, 9, 0]])
    small_graph_g = Graph(small_graph)
    small_graph_g.construct_mst()
    check_mst(small_graph_g.adj_mat, small_graph_g.mst, expected_weight=16)

    # Test with a disconnected graph (should raise an error)
    disconnected_graph = np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, 0]])

    with pytest.raises(ValueError, match="Graph is not connected"):
        Graph(disconnected_graph).construct_mst()

    # Test with a larger graph
    larger_graph = np.random.random((10, 10))
    larger_graph = (larger_graph + larger_graph.T) / 2  # Symmetric
    np.fill_diagonal(larger_graph, 0)  # No self-loops
    larger_graph_g = Graph(larger_graph)
    larger_graph_g.construct_mst()
    check_mst(larger_graph_g.adj_mat, larger_graph_g.mst, expected_weight=np.sum(larger_graph_g.mst) / 2)
