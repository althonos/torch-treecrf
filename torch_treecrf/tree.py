import collections
import typing
from typing import Iterator, Union, Sequence

import torch


class _VariableColumnStorage:
    """A compressed storage for tensors with variable number of columns.
    """

    def __init__(self, data: Sequence[torch.Tensor], device: Union[torch.device, str, None]=None, dtype: Union[torch.dtype, str, None]=None):
        self.indices = torch.zeros(len(data)+1, device=device, dtype=torch.long)
        self.data = torch.zeros(sum(len(row) for row in data), device=device, dtype=dtype)

        for i, row in enumerate(data):
            self.indices[i+1] = self.indices[i] + len(row)
            self.data[self.indices[i]:self.indices[i+1]] = torch.as_tensor(row)

    def __len__(self):
        return self.indices.shape[0] - 1

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return type(self)([self[i] for i in range(*index.indices(len(self)))])
        start = self.indices[index]
        end = self.indices[index+1]
        return self.data[start:end]


class TreeMatrix:
    """A tree encoded as an incidence matrix.
    """

    def __init__(self, data: torch.Tensor, device: Union[torch.device, str, None]=None):
        # store data as sparse matrix
        _data = torch.as_tensor(data, dtype=torch.int32, device=device)
        self.data = _data.to_sparse()

        # cache parents and children
        self._parents = _VariableColumnStorage(
            [torch.where(_data[i, :] != 0)[0] for i in range(_data.shape[0])],
            device=device,
            dtype=torch.long
        )
        self._children = _VariableColumnStorage(
            [torch.where(_data[:, i] != 0)[0] for i in range(_data.shape[0])],
            device=device,
            dtype=torch.long
        )

        # run a BFS walk and store indices
        self._down = self._roots_to_leaves(_data)
        self._up = self._leaves_to_root(_data)

        # store data as a sparse matrix
        self.data = _data.to_sparse()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __iter__(self) -> Iterator[int]:
        return iter(self._down)

    def __reversed__(self) -> Iterator[int]:
        return iter(self._up)

    def _roots_to_leaves(self, data: torch.Tensor) -> torch.Tensor:
        """Generate a walk order to explore the tree from roots to leaves.
        """
        path = -torch.ones(data.shape[0], dtype=torch.long, device=data.device)
        roots = torch.where(data.sum(axis=1) == 0)[0]

        n = 0
        todo = collections.deque(roots)
        done = torch.zeros(data.shape[0], dtype=torch.bool, device=data.device)

        while todo:
            # get next node to visit
            i = todo.popleft()
            # skip if done already
            if done[i]:
                continue
            # delay node visit if we didn't visit some of its children yet
            if not torch.all(done[self.parents(i)]):
                todo.append(i)
                continue
            # add node parents
            todo.extend(self.children(i))
            # mark node as done
            done[i] = True
            path[n] = i
            n += 1

        assert n == len(path)
        assert done.count_nonzero() == data.shape[0]
        return path

    def _leaves_to_root(self, data: torch.Tensor) -> torch.Tensor:
        """Generate a walk order to explore the tree from leaves to roots.
        """
        path = -torch.ones(data.shape[0], dtype=torch.long, device=data.device)
        roots = torch.where(data.sum(axis=0) == 0)[0]

        n = 0
        todo = collections.deque(roots)
        done = torch.zeros(data.shape[0], dtype=torch.bool, device=data.device)

        while todo:
            # get next node to visit
            i = todo.popleft()
            # skip if done already
            if done[i]:
                continue
            # delay node visit if we didn't visit some of its children yet
            if not torch.all(done[self.children(i)]):
                todo.append(i)
                continue
            # add node parents
            todo.extend(self.parents(i))
            # mark node as done
            done[i] = True
            path[n] = i
            n += 1

        assert n == len(path)
        assert done.count_nonzero() == data.shape[0]
        return path

    def parents(self, i: int) -> torch.Tensor:
        return self._parents[i]

    def children(self, i: int) -> torch.Tensor:
        return self._children[i]

    def neighbors(self, i: int) -> torch.Tensor:
        return torch.cat([ self.parents(i), self.children(i) ])