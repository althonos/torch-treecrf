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
from typing import List

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


class TreeCRFLayer(torch.nn.Module):

    def __init__(
        self,
        adjacency: torch.Tensor,
        n_classes: int = 2,
        *,
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

        # `self.pairs[i, j, x_i, x_j]` stores the transitions from
        # class `x_i` of label `i` to class `x_j` of label `j`
        self.pairs = torch.nn.Parameter(
            torch.zeros(
                self.n_labels,
                self.n_labels,
                self.n_classes,
                self.n_classes,
                requires_grad=True,
                **factory_kwargs
            )
        )

        # class hierarchy stored as an adjacency matrix
        self.adjacency = adjacency

        # iteration order and indices from adjacency matrix
        self.parent_labels = _parent_indices(adjacency)
        self.children_labels = _child_indices(adjacency)
        self.downward_order = _downward_walk(adjacency, self.parent_labels, self.children_labels)
        self.upward_order = _upward_walk(adjacency, self.parent_labels, self.children_labels)

    def load_state_dict(self, state, strict=True):
        super().load_state_dict(state, strict=strict)

    def _free_energy(self, X, Y):
        r"""Compute the free energy of :math:`Y` given emissions :math:`X`.

        The free energy of a label vector can be computed with the following
        definition for a conditional random field:

        .. math::

            E(X, Y) = \sum_{i}{ \Psi_i(y_i, x_i) }
                    + \sum_{i, j}{ \Psi_{i, j}(y_i, y_j) }

        such that the conditional probability :math:`P(Y | X)` can be
        expressed as:

        .. math::

            P(Y | X) = \frac{1}{Z(X)} \exp{E(X, Y)}

        where :math:`Z` is the *partition function*, that is computed for
        each emission scores such that probabilities are well-defined for
        the entire event universe.

        Arguments:
            X (torch.Tensor): Input emissions in log-space, as obtained
                from a linear layer, of shape :math:`(*, C, L)` where
                :math:`C` is the number of classes, and :math:`L` the number
                of labels.
            Y (torch.Tensor): Labels vectors, of shape :math:`(*, L)`.

        Returns:
            torch.Tensor: A tensor of shape :math:`(*)` with energy for
            each batch member, in log-space.

        """
        assert X.shape[0] == classes.shape[0]
        assert classes.shape[1] == self.n_labels

        energy = torch.zeros(X.shape[0], device=DEVICE)

        for i in range(self.n_labels):

            # For each i, take log(P(Yi = yi)), whic corresponds to
            energy += X[:, i].gather(1, Y[:, i].unsqueeze(1)).squeeze()

            # For each (i, j) neighbors in the polytree,
            # compute pairwise potential
            # TODO: vectorize?
            neighbors = torch.cat([ self.parents[i], self.children[i] ])
            for j in neighbors:
                energy += self.pairs[i, j, Y[:, i], Y[:, j]]

        return energy

    def _upward_messages(self, emissions):
        r"""Compute messages passed from the leaves to the roots.

        Messages are in log-space for numerical stability. The result in a
        tensor :math:`A` of shape :math:`(*, C, L)` where :math:`*` is the
        batch size, :math:`C` is the number of classes, and :math:`L` the
        number of labels, such that :math:`A[:, i, x_i] = \alpha_i(x_i)` is
        the sum of the messages passed from the children of :math:`i` to
        class :math:`i` for state :math:`x_i`.

        Arguments:
            emissions (`torch.Tensor`): A tensor containing emission scores
                (in log-space) of shape :math:`(n_classes, n_labels)`.

        """
        batch_size, n_classes, n_labels = emissions.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes

        alphas = torch.zeros(emissions.shape[0], self.n_classes, self.n_labels, device=self.pairs.device)
        for j in self.upward_order:
            parents = self.parent_labels[j]
            local = emissions[:, :, j].add(alphas[:, :, j]).unsqueeze(dim=0)
            trans = self.pairs[parents, j].unsqueeze(dim=3).reshape(parents.shape[0], n_classes, 1, n_classes)
            alphas[:, :, parents] += local.add(trans).logsumexp(dim=3).permute(2, 1, 0)

        return alphas

    def _downward_messages(self, emissions):
        r"""Compute messages passed from the roots to the leaves.

        Messages are in log-space for numerical stability. The result in a
        tensor :math:`B` of shape :math:`(*, C, L)` where :math:`*` is the
        batch size, :math:`C` is the number of classes, and :math:`L` the
        number of labels, such that :math:`B[:, i, x_i] = \beta_i(x_i)` is
        the sum of the messages passed from the parents of :math:`i` to
        class :math:`i` for state :math:`x_i`.

        """
        batch_size, n_classes, n_labels = emissions.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes

        betas = torch.zeros(emissions.shape[0], self.n_classes, self.n_labels, device=self.pairs.device)
        for j in self.downward_order:
            children = self.children_labels[j]
            local = emissions[:, :, j].add(betas[:, :, j]).unsqueeze(dim=0)
            trans = self.pairs[children, j].unsqueeze(dim=3).reshape(children.shape[0], n_classes, 1, n_classes)
            betas[:, :, children] += local.add(trans).logsumexp(dim=3).permute(2, 1, 0)

        return betas

    def forward(self, X):
        r"""Compute the marginal probabilities for every label given :math:`X`.

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

        """
        batch_size, n_classes, n_labels = X.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes

        alphas = self._upward_messages(X)
        betas = self._downward_messages(X)

        scores = X + alphas + betas
        logZ = torch.logsumexp(scores, dim=1).unsqueeze(dim=1)
        logP = scores - logZ

        assert torch.all(logP <= 0)
        return logP


class TreeCRF(torch.nn.Module):
    """A Tree-structured CRF for binary classification of labels.

    This implementation uses raw emission scores from a linear layer for
    the node potentials:

    .. math::

        \Psi_i(y_i, x_i) = a^T x_i + b

    This is slightly different from a multiclass problem, where the node
    potentials would be taken as the logarithm of the probability from
    the linear layer, i.e:

    .. math::

        \Psi_i(y_i, x_i) = \log(expit(a^T x_i + b))

    """

    def __init__(
        self,
        in_features: int,
        adjacency: torch.Tensor,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.linear = torch.nn.Linear(in_features, adjacency.shape[0], **factory_kwargs)
        self.crf = TreeCRFLayer(adjacency, **factory_kwargs)

    def forward(self, X):
        emissions_pos = self.linear(X)
        emissions_all = torch.stack((-emissions_pos, emissions_pos), dim=1)
        return self.crf(emissions_all)[:, 1, :].exp()
