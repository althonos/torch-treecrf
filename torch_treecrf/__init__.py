"""A PyTorch implementation of Tree-structured Conditional Random Fields.

References:
    - Tang, Jie, Mingcai Hong, Juanzi Li, and Bangyong Liang.
      "Tree-Structured Conditional Random Fields for Semantic Annotation".
      In The Semantic Web - ISWC 2006, edited by Isabel Cruz, Stefan Decker,
      Dean Allemang, Chris Preist, Daniel Schwabe, Peter Mika, Mike Uschold,
      and Lora M. Aroyo, 640-53. Lecture Notes in Computer Science.
      Berlin, Heidelberg: Springer, 2006. :doi:`10.1007/11926078_46`.

"""

import collections
import typing
from typing import List, Tuple

import torch
from torch import Tensor, LongTensor

__version__ = "0.2.0"
__author__ = "Martin Larralde"
__license__ = "MIT"

__all__ = [
    "TreeCRFLayer",
    "TreeCRF",
]

def _parent_indices(adjacency: Tensor) -> List[LongTensor]:
    return [
        torch.nonzero(adjacency[i, :]).ravel()
        for i in range(adjacency.shape[0])
    ]


def _child_indices(adjacency: Tensor) -> List[LongTensor]:
    return [
        torch.nonzero(adjacency[:, i]).ravel()
        for i in range(adjacency.shape[0])
    ]


def _downward_walk(adjacency: Tensor, parents: List[LongTensor], children: List[LongTensor]) -> LongTensor:
    """Generate a walk order to explore the tree from roots to leaves.
    """
    path = torch.full(adjacency.shape[:1], -1, dtype=torch.long, device=adjacency.device)
    roots = torch.where(adjacency.sum(axis=1) == 0)[0]

    n = 0
    todo = collections.deque(roots)
    done = torch.zeros(adjacency.shape[:1], dtype=torch.bool, device=adjacency.device)

    while todo:
        # get next node to visit
        i = todo.popleft()
        # skip if done already
        if done[i]:
            continue
        # delay node visit if we didn't visit some of its parents yet
        if not torch.all(done[parents[i]]):
            todo.append(i)
            continue
        # add node children
        todo.extend(children[i])
        # mark node as done
        done[i] = True
        path[n] = i
        n += 1

    assert n == len(path)
    assert done.count_nonzero() == adjacency.shape[0]
    return path


def _upward_walk(adjacency: Tensor, parents: List[LongTensor], children: List[LongTensor]) -> LongTensor:
    """Generate a walk order to explore the tree from leaves to roots.
    """
    path = torch.full(adjacency.shape[:1], -1, dtype=torch.long, device=adjacency.device)
    leaves = torch.where(adjacency.sum(axis=0) == 0)[0]

    n = 0
    todo = collections.deque(leaves)
    done = torch.zeros(adjacency.shape[:1], dtype=torch.bool, device=adjacency.device)

    while todo:
        # get next node to visit
        i = todo.popleft()
        # skip if done already
        if done[i]:
            continue
        # delay node visit if we didn't visit some of its parents yet
        if not torch.all(done[children[i]]):
            todo.append(i)
            continue
        # add node children
        todo.extend(parents[i])
        # mark node as done
        done[i] = True
        path[n] = i
        n += 1

    assert n == len(path)
    assert done.count_nonzero() == adjacency.shape[0]
    return path


class MessagePassing(torch.nn.Module):

    def __init__(
        self,
        order: Tensor,
        successors: List[LongTensor],
        *,
        device=None
    ):
        super().__init__()
        # store iteration order
        self.order = order
        # store successors for each label in a compressed sparse tensor
        self.data = torch.concat(successors).to(device=device)
        self.offsets = torch.zeros( len(successors) + 1, dtype=torch.long, device=device )
        for i in range(len(successors)):
            self.offsets[i+1] = self.offsets[i] + len(successors[i])

    def successors(self, i):
        return self.data[self.offsets[i]:self.offsets[i+1]]

    def forward(
        self,
        emissions,
        transitions,
    ):
        batch_size, n_classes, n_labels = emissions.shape
        messages = torch.zeros_like(emissions)
        for j in self.order:
            successors = self.successors(j)
            if successors.size(0) > 0:
                local = emissions[:, :, j]  # shape: (batch_size, n_classes)
                msg = messages[:, :, j]  # shape: (batch_size, n_classes)
                trans = transitions[successors, j]  # shape: (n_succ, n_classes, n_classes)
                elem = local.add(msg).unsqueeze(dim=0).add(trans.unsqueeze(dim=2))
                messages[:, :, successors] += elem.logsumexp(dim=3).permute(2, 1, 0)
        return messages


class _BatchedMessagePassingLayer(torch.nn.Module):

    def __init__(self, layer, successor):
        super().__init__()
        self.register_buffer("layer", layer)
        self.register_buffer("successor", successor)

    def forward(self, emissions: torch.Tensor, transitions: torch.Tensor, messages: torch.Tensor) -> Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ]:
        # extract dimensions
        batch_size, n_classes, n_labels = emissions.shape
        # compute message for all nodes of the current depth
        local = emissions[:, :, self.layer].reshape(1, 1, batch_size, n_classes, -1)
        msg = messages[:, :, self.layer].reshape(1, 1, batch_size, n_classes, -1)
        trans = transitions[:, self.layer].unsqueeze(dim=-2).permute(0, 2, 3, 4, 1)
        elem = local.add(msg).add(trans).logsumexp(dim=3)
        # update the message of parents
        messages += torch.einsum('lkji,il->jkl', elem, self.successor)
        return emissions, transitions, messages


class BatchedMessagePassing(MessagePassing):

    def __init__(
        self,
        order: Tensor,
        successors: List[LongTensor],
        *,
        device=None
    ):
        super().__init__(order, successors, device=device)

        # compute depth of each layer
        self.depths = torch.zeros(len(self.order), dtype=torch.long)
        for i in self.order:
            s = self.successors(i)
            self.depths[s] = torch.max(self.depths[i] + 1, self.depths[s])
        self.max_depth = self.depths.max().item() + 1

        self.layers = torch.nn.ModuleList()

        for depth in range(self.max_depth):
            # compute which labels belong to which layer
            layer = torch.nonzero(self.depths == depth, as_tuple=True)[0]
            # self.register_buffer(f"layer{depth}", layer, persistent=False)

            # compute the indicator for the labels
            s = torch.zeros((len(layer), len(self.order)), device=device )
            for i, j in enumerate(layer):
                s[i, self.successors(j)] = 1.0

            self.layers.append(_BatchedMessagePassingLayer(layer, s))
            # self.register_buffer(f"successor{depth}", s, persistent=False)

        # for


    def forward(
        self,
        emissions,
        transitions,
    ):
        batch_size, n_classes, n_labels = emissions.shape
        messages = torch.zeros_like(emissions)

        for layer in self.layers:
            emissions, transitions, messages = layer(emissions, transitions, messages)

        return messages


class TreeCRFLayer(torch.nn.Module):

    def __init__(
        self,
        adjacency: torch.Tensor,
        n_classes: int = 2,
        *,
        message_passing: str = "layer",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")

        # number of classes and labels
        self.n_labels = adjacency.shape[0]
        self.n_classes = n_classes

        # class hierarchy stored as an adjacency matrix
        self.adjacency = adjacency

        # iteration order and indices from adjacency matrix
        parent_labels = _parent_indices(adjacency)
        children_labels = _child_indices(adjacency)
        downward_order = _downward_walk(adjacency, parent_labels, children_labels)
        upward_order = _upward_walk(adjacency, parent_labels, children_labels)

        # message passing
        self.message_passing = message_passing
        if message_passing == "layer":
            self.alphas = BatchedMessagePassing(upward_order, parent_labels)
            self.betas = BatchedMessagePassing(downward_order, children_labels)
        elif message_passing == "single":
            self.alphas = MessagePassing(upward_order, parent_labels)
            self.betas = MessagePassing(downward_order, children_labels)
        else:
            raise ValueError(f"invalid mode for message passing: {message_passing!r}")

    def load_state_dict(self, state, strict=True):
        super().load_state_dict(state, strict=strict)

    def _upward_messages(self, emissions, transitions):
        r"""Compute messages passed from the leaves to the roots.

        Messages are in log-space for numerical stability. The result in a
        tensor :math:`A` of shape :math:`(*, C, L)` where :math:`*` is the
        batch size, :math:`C` is the number of classes, and :math:`L` the
        number of labels, such that :math:`A[:, i, x_i] = log \alpha_i(x_i)` is
        the sum of the messages passed from the children of :math:`i` to
        class :math:`i` for state :math:`x_i`.

        Arguments:
            emissions (`torch.Tensor`): A tensor containing emission scores
                (in log-space) of shape :math:`(*, C, L)`.
            transitions (`torch.Tensor`): A tensor containing transition
                scores of shape :math:`(L, L, C, C)`.

        """
        batch_size, n_classes, n_labels = emissions.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes
        return self.alphas(emissions, transitions)

    def _downward_messages(self, emissions, transitions):
        r"""Compute messages passed from the roots to the leaves.

        Messages are in log-space for numerical stability. The result in a
        tensor :math:`B` of shape :math:`(*, C, L)` where :math:`*` is the
        batch size, :math:`C` is the number of classes, and :math:`L` the
        number of labels, such that :math:`B[:, i, x_i] = log \beta_i(x_i)` is
        the sum of the messages passed from the parents of :math:`i` to
        class :math:`i` for state :math:`x_i`.

        Arguments:
            emissions (`torch.Tensor`): A tensor containing emission scores
                (in log-space) of shape :math:`(*, C, L)`.
            transitions (`torch.Tensor`): A tensor containing transition
                scores of shape :math:`(L, L, C, C)`.

        """
        batch_size, n_classes, n_labels = emissions.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes
        return self.betas(emissions, transitions)

    def forward(self, emissions, transitions):
        r"""Compute marginal log-probabilities for every label.

        By definition, in a CRF the conditional probability for is defined as:

        .. math::

            p(Y | X) = \frac1Z
                       \Prod_{i=1}^n{\Psi_i(y_i, x_i)}
                       \Prod_{j \in \mathcal{N}(i)}

        where :math:`\mathcal{N}(i)` is set of neighbours of label :math:`i`
        in the tree.

        Marginal probabilities for label :math:`y_i` are obtained by
        summing over conditional probabilities. Once factoring the messages
        passed over the model graph with the *belief propagation*, we get:

        .. math::

            p(y_i | x_i) = \frac1Z \Psi_i(y_i, x_i)
                           \Prod_{j in \mathcal{N}(i) \mu_{j \to i}(y_i)}

        where message from children to parents are defined by recurence,
        :math:`\forall i \in \{ 1, .., n \} , \forall j \in \mathcal{C}(i)`,

        .. math::

            \mu_{j \to i}(y_i) = \sum_{y_j}{
                \Psi_{i,j}(y_i, y_j)
                \Psi_j(y_j, x_j)
                \Prod_{k \in \mathcal{C}(j)}{ \mu_{k \to j}(y_j) }
            }

        (and conversely for messages from parents to children). This
        reccurence relationship is known as the Sum Product Algorithm.

        The partition function is computed so that the probabilities are
        well defined, i.e. that they sum to one for every label of class
        :math:`i`:

        .. math::

            Z = \sum_{y_i}{
                \Psi_i(y_i, x_i)
                \Prod_{j in \mathcal{N}(i)}{\mu_{j \to i}(y_i)}
            }

        In log-space, the marginals can be expressed as a sum:

        .. math::

            \log p(y_i | x_i) = \log \Psi_i(y_i, x_i)
                                - \log Z
                                + \sum_{j in \mathcal{N}(i)}{
                                    \log \mu_{j \to i}(y_i)
                                }

        For a tree CRF in particular, the last term can be decomposed in two
        components: the sum of messages passed from the parents, and the sum
        of messages passed from the children:

        .. math::

            \begin{aligned}
                \sum_{j in \mathcal{N}(i)}{ \log \mu_{j \to i}(y_i) }
             & = \sum_{j in \mathcal{C}(i)}{ \log \mu_{j \to i}(y_i) }
             & + \sum_{j in \mathcal{P}(i)}{ \log \mu_{j \to i}(y_i) } \\
             & = \Alpha_i(y_i) & + \Beta_i(y_i)
            \end{aligned}

        :math:`\Alpha` and :math:`\Beta` can be computed efficiently with
        the forward-backward algorithm, and stored in a tensor of shape
        :math:`(*, C, L)`.

        Arguments:
            emissions (`torch.tensor` of shape $(*, C, L)$): The values
                of unary feature functions for the batch, i.e
                :math:`\f_i(y_i, x_i)` used to compute the potentials
                :math:`\Psi_i(y_i, x_i) = exp(\sum{\theta \f_i(y_i, x_i)})`.
            transitions (`torch.tensor` of shape $(L, L, C, C)$): The
                values of binary feature functions, i.e.
                :math:`\f_{i,j}(y_i, y_j)` used to compute the potentials
                :math:`\Psi_{i,j}(y_i, y_j) = exp(\sum{\theta \f_{i,j}(y_i, y_j)})`.

        """
        batch_size, n_classes, n_labels = emissions.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes

        alphas = self._upward_messages(emissions, transitions)
        betas = self._downward_messages(emissions, transitions)

        scores = emissions + alphas + betas
        logZ = torch.logsumexp(scores, dim=1).unsqueeze(dim=1)
        logP = scores - logZ

        return logP


class TreeCRF(torch.nn.Module):
    r"""A Tree-structured CRF for binary classification of labels.

    This implementation uses unary feature scores from a previous layer and
    learns the binary feature scores.

    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        *,
        message_passing: str = "layer",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.crf = TreeCRFLayer(
            adjacency,
            n_classes=2,
            message_passing=message_passing,
            **factory_kwargs
        )
        self.transitions = torch.nn.Parameter(
            torch.ones(
                adjacency.shape[0],
                adjacency.shape[0],
                2,
                2,
                requires_grad=True,
                **factory_kwargs,
            )
        )

    def forward(self, X):
        batch_size, n_labels = X.shape
        assert n_labels == self.crf.n_labels
        emissions = torch.stack((-X, X), dim=1)
        logp = self.crf(emissions, self.transitions)
        return logp[:, 1, :] - logp[:, 0, :]
