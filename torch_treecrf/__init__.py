import torch

from .tree import TreeMatrix


__version__ = "0.0.1"
__author__ = "Martin Larralde"
__license__ = "MIT"


class TreeCRFLayer(torch.nn.Module):

    def __init__(
        self, 
        labels: TreeMatrix, 
        n_classes: int = 2, 
        device=None, 
        dtype=None
    ):
        super().__init__()

        # number of classes and labels
        self.n_labels = len(labels)
        self.n_classes = n_classes

        # class hierarchy stored as an incidence matrix 
        self.labels = labels
        self.labels_data = torch.nn.Parameter(labels.data, requires_grad=False)

        # `self.pairs[i, j, x_i, x_j]` stores the transitions from 
        # class `x_i` of label `i` to class `x_j` of label `j`
        self.pairs = torch.nn.Parameter(
            torch.empty(
                self.n_labels, 
                self.n_labels, 
                self.n_classes, 
                self.n_classes, 
                requires_grad=True, 
                device=device,
                dtype=dtype or torch.float32,
            )
        )
        torch.nn.init.uniform_(self.pairs, 0.1, 1.0)

    def load_state_dict(self, state, strict=True):
        super().load_state_dict(state, strict=strict)
        self.labels = TreeMatrix(self.labels_data)

    def _free_energy(self, emissions, classes):
        r"""Compute the free energy of the class labeling given emissions.

        The free energy of a conditional random field can be computed
        with the following definition:

        .. math::

            E(X, Y) = \sum_{i}{ \Psi_i(y_i, x_i) } + \sum_{i, j}{ \Psi_{i, j}(y_i, y_j) }

        such that the conditional probability :math:`P(Y | X)` can be
        expressed as:

        .. math::

            P(Y | X) = \frac{1}{Z(X)} \exp{E(X, Y)}

        where :math:`Z` is the *partition function*, that is computed for
        each emission scores such that probabilities are well-defined for 
        the entire event universe.

        Arguments:
            emissions (torch.Tensor): Input emissions in log-space, as obtained
                from a linear layer. Shape :math:`(*, C, K)` where :math:`C`
                is the number of classes, and :math:`K` the number of labels.
            Y (torch.Tensor): Class labels. Shape :math:`(*, C)`.

        Returns:
            torch.Tensor: A tensor of shape :math:`(*)` with energy for
            each batch member, in log-space.

        """
        assert emissions.shape[0] == classes.shape[0]
        assert classes.shape[1] == len(self.hierarchy)

        energy = torch.zeros( emissions.shape[0], device=DEVICE)

        for i in range(len(self.hierarchy)):

            # For each i, take log(P(Yi = yi)), whic corresponds to
            # emissions[i] if y_i = 0 and -emissions[i] otherwise.
            energy += emissions[:, i].gather(1, classes[:, i].unsqueeze(1)).squeeze()

            # For each (i, j) neighbors in the polytree,
            # compute pairwise potential
            # TODO: vectorize?
            for j in self.hierarchy.neighbors(i):
                energy += self.pairs[i, j, classes[:, i], classes[:, j]]

        return energy

    def _upward_messages(self, emissions):
        r"""Compute messages passed from the leaves to the roots.

        Messages are in log-space for numerical stability. The result in a
        tensor :math:`A` of shape :math:`(*, C, L)` where :math:`*` is the 
        batch size, :math:`C` is the number of classes, and :math:`L` the 
        number of labels, such that :math:`A[:, i, x_i] = \Alpha_i(x_i)` is
        the sum of the messages passed from the children of :math:`i` to 
        class :math:`i` for state :math:`x_i`.

        Arguments:
            emissions (`torch.Tensor`): A tensor containing emission scores
                (in log-space) of shape :math:`(n_classes, n_labels)`.

        """
        batch_size, n_labels, n_classes = emissions.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes

        alphas = torch.zeros( emissions.shape[0], self.n_labels, self.n_classes, device=self.pairs.device)

        # iterate from leaves to root
        # for obs in range(emissions.shape[0]):
        #     for j in reversed(self.hierarchy):
        #         j_children = self.hierarchy.children(j)
        #         for i in self.hierarchy.parents(j):
        #             for x_i in range(self.n_labels):
        #                 scores = self.pairs[i, j, x_i, :] + emissions[obs, j, :]
        #                 assert scores.shape == ( self.n_labels, )
        #                 for k in j_children:
        #                     scores += messages[obs, k, j]
        #                 messages[obs, j, i, x_i] = torch.logsumexp(scores, 0)

        # for j in reversed(self.hierarchy):
        #     j_children = self.hierarchy.children(j)
        #     for i in self.hierarchy.parents(j):
        #         for x_i in range(self.n_labels):
        #             scores = self.pairs[i, j, x_i, :] + emissions[:, j, :]
        #             assert scores.shape == ( emissions.shape[0], self.n_labels, )
        #             for k in j_children:
        #                 scores += messages[:, k, j]
        #             messages[:, j, i, x_i] = torch.logsumexp(scores, 1)

        for j in reversed(self.labels):
            local_scores = emissions[:, j, :] + alphas[:, j, :]
            for i in self.labels.parents(j):
                trans_scores = self.pairs[i, j, :, :].repeat(1, 1, batch_size).reshape(n_classes, batch_size, n_classes)
                alphas[:, i, :] += torch.logsumexp(local_scores + trans_scores, dim=2).T

        return alphas

    def _downward_messages(self, emissions):
        r"""Compute messages passed from the roots to the leaves.

        Messages are in log-space for numerical stability. The result in a
        tensor :math:`B` of shape :math:`(*, L, C)` where :math:`*` is the 
        batch size, :math:`C` is the number of classes, and :math:`L` the 
        number of labels, such that :math:`B[:, i, x_i] = \Beta_i(x_i)` is
        the sum of the messages passed from the parents of :math:`i` to 
        class :math:`i` for state :math:`x_i`.

        """
        batch_size, n_labels, n_classes = emissions.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes

        betas = torch.zeros( emissions.shape[0], self.n_labels, self.n_classes, device=self.pairs.device)

        # iterate from root to leaves
        # for obs in range(emissions.shape[0]):
        #     for j in self.hierarchy:
        #         j_parents = self.hierarchy.parents(j)
        #         for i in self.hierarchy.children(j):
        #             for x_i in range(self.n_labels):
        #                 scores = self.pairs[i, j, x_i, self.labels] + emissions[obs, j, self.labels]
        #                 assert scores.shape == ( self.n_labels, )
        #                 for k in j_parents:
        #                     scores += messages[obs, k, j]
        #                 messages[obs, j, i, x_i] = torch.logsumexp(scores, 0)

        for j in self.labels:
            local_scores = emissions[:, j, :] + betas[:, j, :]
            for i in self.labels.children(j):
                trans_scores = self.pairs[i, j, :, :].repeat(1, 1, batch_size).reshape(n_classes, batch_size, n_classes)
                betas[:, i, :] += torch.logsumexp(local_scores + trans_scores, dim=2).T

        return betas

    def forward(self, X):
        r"""Compute the marginal probabilities for every label given :math:`X`.

        By definition, in a CRF the conditional probability for is defined as:

        .. math::

            p(Y | X) = \frac1Z \Prod_{i=1}^n{\Psi_i(y_i, x_i)} \Prod_{j \in \mathcal{N}(i)}

        where :math:`\mathcal{N}(i)` is set of neighbours of label :math:`i`
        in the tree.

        Marginal probabilities for label :math:`y_i` are obtained by 
        summing over conditional probabilities. Once factoring the messages
        passed over the model graph with the *belief propagation*, we get:

        .. math::

            p(y_i | x_i) = \frac1Z \Psi_i(y_i, x_i) \Prod_{j in \mathcal{N}(i) \mu_{j \to i}(y_i)}

        where message from children to parents are defined by recurence, 
        :math:`\forall i \in \{ 1, .., n \} , \forall j \in \mathcal{C}(i)`, 

        .. math::

            \mu_{j \to i}(y_i) = \sum_{y_j}{ \Psi_{i,j}(y_i, y_j) \Psi_j(y_j, x_j) \Prod_{k \in \mathcal{C}(j)}{ \mu_{k \to j}(y_j) } }

        (and conversely for messages from parents to children).

        The partition function is computed so that the probabilities are
        well defined, i.e. that they sum to one for all values of :math:`y_i`:

        .. math::

            Z = \sum_{y_i} \Psi_i(y_i, x_i) \Prod_{j in \mathcal{N}(i) \mu_{j \to i}(y_i)}

        In log-space, the marginals can be expressed as a sum:

        .. math::

            \log p(y_i | x_i) = \log \Psi_i(y_i, x_i) - \log Z + \sum_{j in \mathcal{N}(i)}{ \log \mu_{j \to i}(y_i) }
        
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
        :math:`(*, L, C)`.

        """
        batch_size, n_labels, n_classes = X.shape
        assert n_labels == self.n_labels
        assert n_classes == self.n_classes

        alphas = self._upward_messages(X)  
        betas = self._downward_messages(X)

        scores = X + alphas + betas
        logZ = torch.logsumexp(scores, 2).unsqueeze(2)
        logP = scores - logZ

        assert torch.all(logP <= 0)
        return logP


class TreeCRF(torch.nn.Module):

    def __init__(
        self, 
        n_features: int, 
        hierarchy: TreeMatrix,
        device=None,
        dtype=None,
    ):  
        super().__init__()
        self.linear = torch.nn.Linear(n_features, len(hierarchy), device=device, dtype=dtype)
        self.crf = TreeCRFLayer(hierarchy, device=device, dtype=dtype)

    def forward(self, X):
        emissions = self.linear(X)
        logP = self.crf(torch.stack((-emissions, emissions)))
        return torch.exp(logP)[:, 1]








    