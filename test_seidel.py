import numpy as np
import pytest

from seidel_algo import seidel_algorithm


def test_fully_connected():
    A = [[0, 1, 1, 1],
         [1, 0, 1, 1],
         [1, 1, 0, 1],
         [1, 1, 1, 0]]
    res = A.copy()
    assert np.array_equal(seidel_algorithm(A), res)


def test_all_zeros():
    A = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    res = [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]
    assert np.array_equal(seidel_algorithm(A), res)


def test_simple_graph_4x4():
    A = [[0, 1, 1, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 1],
         [0, 0, 1, 0]]
    res = [[0, 1, 1, 2],
           [1, 0, 2, 3],
           [1, 2, 0, 1],
           [2, 3, 1, 0]]
    assert np.array_equal(seidel_algorithm(A), res)


def test_simple_graph_6x6():
    A = [[0, 1, 0, 0, 1, 1],
         [1, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0],
         [1, 0, 0, 1, 0, 0],
         [1, 0, 1, 0, 0, 0]]
    res = symmetrize([
        [0, 1, 2, 2, 1, 1],
        [0, 0, 1, 3, 2, 2],
        [0, 0, 0, 4, 3, 1],
        [0, 0, 0, 0, 1, 3],
        [0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0]])
    assert np.array_equal(seidel_algorithm(A), res)


def test_disconnected_graph_7x7():
    A = [
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0]]
    res = [
        [0, 1, 2, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0],
        [2, 1, 0, 2, 0, 0, 0],
        [1, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 2, 1, 0]]
    assert np.array_equal(seidel_algorithm(A), res)


def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.
    """
    a = np.asarray(a)
    return a + a.T - np.diag(a.diagonal())


if __name__ == '__main__':
    pytest.main()
