import collections

import torch


class TreeMatrix:
    """A tree encoded as an incidence matrix.
    """

    def __init__(self, data: torch.Tensor, device=None):
        # store data
        self.data = data

        # get the index of the root
        self.roots = torch.where(data.sum(axis=1) == 0)[0]
        if len(self.roots) == 0:
            raise ValueError(f"Failed to find roots in tree: {self.roots}")

        # cache parents and children
        self._parents = []
        self._children = []
        for i in range(len(self)):
            self._parents.append( torch.where(self.data[i, :] != 0)[0] )
            self._children.append( torch.where(self.data[:, i] != 0)[0] )

        # run a BFS walk and store indices
        done = set()
        self.bfs_path = -torch.ones(self.data.shape[0]+2, dtype=torch.int32, device=device)
        n = 0
        q = collections.deque(self.roots)
        while q:
            node = q.popleft()
            q.extend(self.children(node))
            self.bfs_path[n] = node
            n += 1

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        return iter(self.bfs_path)

    def __reversed__(self):
        return reversed(self.bfs_path)

    def parents(self, i):
        # return torch.where(self.data[i, :] != 0)[0]
        return self._parents[i]

    def children(self, i):
        # return torch.where(self.data[:, i] != 0)[0]
        return self._children[i]

    def neighbors(self, i):
        return torch.cat([ self.parents(i), self.children(i) ])


class TreeCRFLayer(torch.nn.Module):

    def __init__(self, hierarchy: TreeMatrix, n_labels: int = 2, device=None):
        super().__init__()

        # number of classes and labels
        self.n_labels = n_labels
        self.n_classes = len(hierarchy)

        # class hierarchy stored as an incidence matrix 
        self.hierarchy = hierarchy
        self.hierarchy_data = torch.nn.Parameter(hierarchy.data, requires_grad=False)

        # `self.pairs[i, j, x_i, x_j]` stores the transitions from class
        # `i`` in state `x_i` to class `j` in state `x_j`.
        self.pairs = torch.nn.Parameter(
            torch.ones(
                self.n_classes, 
                self.n_classes, 
                self.n_labels, 
                self.n_labels, 
                requires_grad=True, 
                device=device
            )
        )
        torch.nn.init.uniform_(self.pairs, 0.1, 1.0)

    def load_state_dict(self, state, strict=True):
        super().load_state_dict(state, strict=strict)
        self.hierarchy = TreeMatrix(self.hierarchy_data)

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
        batch_size, n_classes, n_labels = emissions.shape
        alphas = torch.zeros( emissions.shape[0], self.n_classes, self.n_labels, device=self.pairs.device)

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

        for j in reversed(self.hierarchy):
            local_scores = emissions[:, j, :] + alphas[:, j, :]
            for i in self.hierarchy.parents(j):
                trans_scores = self.pairs[i, j, :, :].repeat(1, 1, batch_size).reshape(n_labels, batch_size, n_labels)
                alphas[:, i, :] += torch.logsumexp(local_scores + trans_scores, dim=2).T

        return alphas

    def _downward_messages(self, emissions):
        r"""Compute messages passed from the roots to the leaves.

        Messages are in log-space for numerical stability. The result in a
        tensor :math:`B` of shape :math:`(*, C, L)` where :math:`*` is the 
        batch size, :math:`C` is the number of classes, and :math:`L` the 
        number of labels, such that :math:`B[:, i, x_i] = \Beta_i(x_i)` is
        the sum of the messages passed from the parents of :math:`i` to 
        class :math:`i` for state :math:`x_i`.

        """
        batch_size, n_classes, n_labels = emissions.shape
        betas = torch.zeros( emissions.shape[0], self.n_classes, self.n_labels, device=self.pairs.device)

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

        for j in self.hierarchy:
            local_scores = emissions[:, j, :] + betas[:, j, :]
            for i in self.hierarchy.children(j):
                trans_scores = self.pairs[i, j, :, :].repeat(1, 1, batch_size).reshape(n_labels, batch_size, n_labels)
                betas[:, i, :] += torch.logsumexp(local_scores + trans_scores, dim=2).T

        return betas

    def forward(self, X):
        r"""Compute the marginal probabilities for every class given :math:`X`.

        By definition, in a CRF the conditional probability for is defined as:

        .. math::

            p(Y | X) = \frac1Z \Prod_{i=1}^n{\Psi_i(y_i, x_i)} \Prod_{j \in \mathcal{N}(i)}

        where :math:`\mathcal{N}(i)` is set of neighbours of class :math:`i`
        in the graph.

        Marginal probabilities for observation :math:`y_i` are obtained by 
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
        :math:`(*, C, L)`.

        """
        alphas = self._upward_messages(X)  
        betas = self._downward_messages(X)

        logits = X + alphas + betas
        logZ = torch.logsumexp(logits, 2).unsqueeze(2)
        logP = logits - logZ

        assert torch.all(logP <= 0)
        return logP


class TreeCRF(torch.nn.Module):

    def __init__(self, n_features: int, hierarchy: TreeMatrix):  
        super().__init__()
        self.linear = torch.nn.Linear(n_features, len(hierarchy))
        self.crf = TreeCRFLayer(hierarchy, n_labels=2)

    def forward(self, X):
        emissions = self.linear(X)
        logP = self.crf(torch.stack((-emissions, emissions)))
        return torch.exp(logP)[:, 1]








    