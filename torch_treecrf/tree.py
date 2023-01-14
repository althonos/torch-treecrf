import collections
import typing
from typing import Iterator, Union

import torch


class TreeMatrix:
    """A tree encoded as an incidence matrix.
    """

    def __init__(self, data: torch.Tensor, device: Union[torch.device, str, None]=None):
        # store data
        _data = torch.as_tensor(data, dtype=torch.int32, device=device)

        # get the index of the root
        self.roots = torch.where(_data.sum(axis=1) == 0)[0]
        if len(self.roots) == 0:
            raise ValueError(f"Failed to find roots in tree: {self.roots}")

        # cache parents and children
        self._parents = []
        self._children = []
        for i in range(len(self)):
            self._parents.append( torch.where(_data[i, :] != 0)[0] )
            self._children.append( torch.where(_data[:, i] != 0)[0] )

        # run a BFS walk and store indices
        done = set()
        self.bfs_path = -torch.ones(_data.shape[0], dtype=torch.int32, device=device)
        n = 0
        q = collections.deque(self.roots)
        while q:
            node = q.popleft()
            q.extend(self.children(node))
            self.bfs_path[n] = node
            n += 1

        # store data as a sparse matrix
        self.data = _data.to_sparse()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __iter__(self) -> Iterator[int]:
        return iter(self.bfs_path)

    def __reversed__(self) -> Iterator[int]:
        return reversed(self.bfs_path)

    def parents(self, i: int) -> torch.Tensor:
        return self._parents[i]

    def children(self, i: int) -> torch.Tensor:
        return self._children[i]

    def neighbors(self, i: int) -> torch.Tensor:
        return torch.cat([ self.parents(i), self.children(i) ])