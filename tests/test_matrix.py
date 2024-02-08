import unittest

import torch
from torch_treecrf import _parent_indices, _child_indices



class TestParents(unittest.TestCase):

    def test_fence(self):
        """Check that path algorithms work on a fence tree.
        """
        matrix = torch.tensor([
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 1, 0, 1 ],
            [ 0, 0, 0, 0, 0 ],
        ])

        parents = _parent_indices(matrix)
        self.assertEqual(list(parents[0]), [])
        self.assertEqual(list(parents[1]), [0, 2])
        self.assertEqual(list(parents[2]), [])
        self.assertEqual(list(parents[3]), [2, 4])
        self.assertEqual(list(parents[4]), [])

    def test_comb(self):
        """Check that path algorithms work on a comb tree.
        """
        matrix = torch.tensor([
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 0, 0, 1, 0, 0 ],
            [ 0, 0, 1, 0, 0 ],
        ])

        parents = _parent_indices(matrix)
        self.assertEqual(list(parents[0]), [])
        self.assertEqual(list(parents[1]), [0])
        self.assertEqual(list(parents[2]), [0])
        self.assertEqual(list(parents[3]), [2])
        self.assertEqual(list(parents[4]), [2])
  

class TestChildren(unittest.TestCase):

    def test_fence(self):
        """Check that path algorithms work on a fence tree.
        """
        matrix = torch.tensor([
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 1, 0, 1 ],
            [ 0, 0, 0, 0, 0 ],
        ])

        children = _child_indices(matrix)
        self.assertEqual(list(children[0]), [1])
        self.assertEqual(list(children[1]), [])
        self.assertEqual(list(children[2]), [1, 3])
        self.assertEqual(list(children[3]), [])
        self.assertEqual(list(children[4]), [3])

    def test_comb(self):
        """Check that path algorithms work on a comb tree.
        """
        matrix = torch.tensor([
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 0, 0, 1, 0, 0 ],
            [ 0, 0, 1, 0, 0 ],
        ])

        children = _child_indices(matrix)
        self.assertEqual(list(children[0]), [1, 2])
        self.assertEqual(list(children[1]), [])
        self.assertEqual(list(children[2]), [3, 4])
        self.assertEqual(list(children[3]), [])
        self.assertEqual(list(children[4]), [])
