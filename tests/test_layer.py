import unittest

import torch.nn
import torch.jit
from torch_treecrf import TreeCRFLayer



class TestTreeCRFLayer(unittest.TestCase):

    def test_partition_function(self):
        """Check that `TreeCRFLayer` properly computes the partition function.
        """

        adj = torch.tensor([[ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 0, 0 ]])
        layer = TreeCRFLayer(adj, n_classes=5)

        # mock emission scores generated from a linear layer
        base_probas = torch.rand(100, layer.n_classes, layer.n_labels)
        base_probas /= base_probas.sum(dim=2).unsqueeze(2)
        emissions = torch.log(base_probas)

        # mock transition scores
        transitions = torch.ones((3, 3, 5, 5))

        # check that the CRF computes probabilities that sum to 1.0 for all labels
        probas = torch.exp(layer(emissions, transitions)) 
        for i in range(emissions.shape[0]):
            for j in range(layer.n_labels):
                self.assertAlmostEqual(torch.sum(probas[i, :, j]).item(), 1.0, places=3)


    def test_script(self):
        adj = torch.tensor([[ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 0, 0 ]])
        layer = TreeCRFLayer(adj, n_classes=5, device="cpu")

        scripted_layer = torch.jit.script(layer)

        # mock emission scores generated from a linear layer
        base_probas = torch.rand(100, layer.n_classes, layer.n_labels)
        base_probas /= base_probas.sum(dim=2).unsqueeze(2)
        emissions = torch.log(base_probas)

        # mock transition scores
        transitions = torch.ones((3, 3, 5, 5))

        probas = torch.exp(scripted_layer(emissions, transitions))
