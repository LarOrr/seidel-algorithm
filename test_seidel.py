import pytest
# import unittest
import numpy as np
import seidel_algo

def test_simple_matrix():
    A = [[0, 1, 1, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 1],
         [0, 0, 1, 0]]
    res = [[0, 1, 1, 2],
           [1, 0, 2, 3],
           [1, 2, 0, 1],
           [2, 3, 1, 0]]
    assert np.array_equal(seidel_algo.seidel_algorithm(A), res)

# class TestSum(unittest.TestCase):
#     def test_simple_matrix(self):
#         A = [[0, 1, 1, 0],
#             [1, 0, 0, 0],
#             [1, 0, 0, 1],
#             [0, 0, 1, 0]]
#         res = [[0, 1, 1, 2],
#             [1, 0, 2, 3],
#             [1, 2, 0, 1],
#             [2, 3, 1, 0]]
#         self.assertTrue(np.array_equal(seidel_algo.seidel_algo(A), res))

# if __name__ == '__main__':
#     unittest.main()

if __name__ == '__main__':
    pytest.main()