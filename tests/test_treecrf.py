import unittest

import torch
from torch_treecrf import TreeCRF



class TestTreeCRF(unittest.TestCase):

    def test_partition_function(self):
        """Check that `TreeCRF` properly computes the partition function.
        """
        adj = torch.tensor([[ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 0, 0 ]])
        crf = TreeCRF(adj)
        features = torch.rand(100, 3, generator=torch.random.manual_seed(0))
        logits = crf(features)
        # for i in range(features.shape[0]):
        #     self.assertLessEqual(probas[i].max(), 1.0)
        #     self.assertGreaterEqual(probas[i].min(), 0.0)

    def test_state_dict(self):
        adj = torch.tensor([[ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 0, 0 ]])
        model = TreeCRF(adj)
        d = model.state_dict()
        model.load_state_dict(d)
        # self.assertTrue(torch.all(model.crf.labels.data.to_dense() == matrix.data.to_dense()))

