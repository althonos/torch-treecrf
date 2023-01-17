import unittest

import torch
from torch_treecrf import TreeMatrix



class TestTreeMatrix(unittest.TestCase):

    def test_fence(self):
        """Check that `TreeMatrix` works on a fence-structured hierarchy graph.
        """
        matrix = TreeMatrix([
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 1, 0, 1 ],
            [ 0, 0, 0, 0, 0 ],
        ])

        self.assertEqual(list(matrix.parents(0)), [])
        self.assertEqual(list(matrix.parents(1)), [0, 2])
        self.assertEqual(list(matrix.parents(2)), [])
        self.assertEqual(list(matrix.parents(3)), [2, 4])
        self.assertEqual(list(matrix.parents(4)), [])

        self.assertEqual(list(matrix.children(0)), [1])
        self.assertEqual(list(matrix.children(1)), [])
        self.assertEqual(list(matrix.children(2)), [1, 3])
        self.assertEqual(list(matrix.children(3)), [])
        self.assertEqual(list(matrix.children(4)), [3])

    def test_comb(self):
        """Check that `TreeMatrix` works on a comb-structured hierarchy graph.
        """
        matrix = TreeMatrix([
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 0, 0, 1, 0, 0 ],
            [ 0, 0, 1, 0, 0 ],
        ])

        self.assertEqual(list(matrix.parents(0)), [])
        self.assertEqual(list(matrix.parents(1)), [0])
        self.assertEqual(list(matrix.parents(2)), [0])
        self.assertEqual(list(matrix.parents(3)), [2])
        self.assertEqual(list(matrix.parents(4)), [2])

        self.assertEqual(list(matrix.children(0)), [1, 2])
        self.assertEqual(list(matrix.children(1)), [])
        self.assertEqual(list(matrix.children(2)), [3, 4])
        self.assertEqual(list(matrix.children(3)), [])
        self.assertEqual(list(matrix.children(4)), [])

  

