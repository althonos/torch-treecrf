import unittest

import torch
from torch_treecrf import TreeMatrix, TreeCRFLayer



class TestTreeCRFLayer(unittest.TestCase):

    def test_partition_function(self):
        """Check that `TreeCRFLayer` properly computes the partition function.
        """

        matrix = TreeMatrix([[ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 0, 0 ]])
        layer = TreeCRFLayer(matrix, n_classes=5)

        # mock emission scores generated from a linear layer
        base_probas = torch.rand(100, layer.n_classes, layer.n_labels)
        base_probas /= base_probas.sum(dim=2).unsqueeze(2)
        emissions = torch.log(base_probas)

        # check that the CRF computes probabilities that sum to 1.0 for all labels
        probas = torch.exp(layer(emissions)) 
        for i in range(emissions.shape[0]):
            for j in range(layer.n_labels):
                self.assertAlmostEqual(torch.sum(probas[i, :, j]).item(), 1.0, places=3)


  

