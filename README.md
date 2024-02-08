# 🌲 `torch-treecrf`

*A [PyTorch](https://pytorch.org/) implementation of Tree-structured Conditional Random Fields.*

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/torch-treecrf/test.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/torch-treecrf/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/torch-treecrf?style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/torch-treecrf/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/gpl-3.0/)
[![PyPI](https://img.shields.io/pypi/v/torch-treecrf.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/torch-treecrf)
[![Wheel](https://img.shields.io/pypi/wheel/torch-treecrf.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/torch-treecrf/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/torch-treecrf.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/torch-treecrf/#files)
[![Python Implementations](https://img.shields.io/badge/impl-universal-success.svg?style=flat-square&maxAge=3600&label=impl)](https://pypi.org/project/torch-treecrf/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/torch-treecrf/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/torch-treecrf.svg?style=flat-square&maxAge=600)](https://github.com/althonos/torch-treecrf/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/torch-treecrf.py/blob/master/CHANGELOG.md)
[![Downloads](https://img.shields.io/badge/dynamic/json?style=flat-square&color=303f9f&maxAge=86400&label=downloads&query=%24.total_downloads&url=https%3A%2F%2Fapi.pepy.tech%2Fapi%2Fprojects%2Ftorch-treecrf)](https://pepy.tech/project/torch-treecrf)

## 🗺️ Overview

[Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field)
(CRF) are a family of discriminative graphical learning models that can be used
to model the dependencies between variables. The most common
form of CRFs are Linear-chain CRF, where a prediction depends on
an observed variable, as well as the prediction before and after it
(the *context*). Linear-chain CRFs are widely used in Natural Language Processing.

<p align="center">
  <img height="150" src="https://github.com/althonos/torch-treecrf/raw/main/static/linear-chain-crf.svg?raw=true">
</p>

$$
P(Y | X) = \frac{1}{Z(X)} \prod_{i=1}^n{ \Psi_i(y_i, x_i) } \prod_{i=2}^n{ \Psi_{i-1,i}(y_{i-1}, y_i)}
$$

In 2006, Tang *et al.*[[1]](#ref1) introduced Tree-structured CRFs to model hierarchical
relationships between predicted variables, allowing dependencies between
a prediction variable and its parents and children.

<p align="center">
  <img height="280" src="https://github.com/althonos/torch-treecrf/raw/main/static/tree-structured-crf.svg?raw=true">
</p>

$$
P(Y | X) = \frac{1}{Z(X)} \prod_{i=1}^{n}{ \Psi_i(y_i, x_i) } \prod_{j \in \mathcal{N}(i)}{ \Psi_{j,i}(y_j, y_i)}
$$

This package implements a generic Tree-structured CRF layer in PyTorch. The
layer can be stacked on top of a [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) to implement a proper Tree-structured CRF, or on any other kind of model
producing emission scores in log-space for every class of each label. Computation
of marginals is implemented using Belief Propagation[[2]](#ref2), allowing for
exact inference on trees[[3]](#ref3):

$$
\begin{aligned}
P(y_i | X)
& =
    \frac{1}{Z(X)} \Psi_i(y_i, x_i)
    & \underbrace{\prod_{j \in \mathcal{C}(i)}{\mu_{j \to i}(y_i)}} &
    & \underbrace{\prod_{j \in \mathcal{P}(i)}{\mu_{j \to i}(y_i)}} \\
& = \frac1Z \Psi_i(y_i, x_i)
    & \alpha_i(y_i) &
    & \beta_i(y_i)  \\
\end{aligned}
$$

where for every node $i$, the message from the parents $\mathcal{P}(i)$ and
the children $\mathcal{C}(i)$ is computed recursively with the sum-product algorithm[[4]](#ref4):

$$
\begin{aligned}
\forall j \in \mathcal{C}(i), \mu_{j \to i}(y_i) = \sum_{y_j}{
  \Psi_{i,j}(y_i, y_j)
  \Psi_j(y_j, x_j)
  \prod_{k \in \mathcal{C}(j)}{\mu_{k \to j}(y_j)}
} \\
\forall j \in \mathcal{P}(i), \mu_{j \to i}(y_i) = \sum_{y_j}{
  \Psi_{i,j}(y_i, y_j)
  \Psi_j(y_j, x_j)
  \prod_{k \in \mathcal{P}(j)}{\mu_{k \to j}(y_j)}
} \\
\end{aligned}
$$


*The implementation should be generic enough that any kind of [Directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) can be used as a label hierarchy,
not just trees.*

## 🔧 Installing

Install the `torch-treecrf` package directly from [PyPi](https://pypi.org/project/peptides)
which hosts universal wheels that can be installed with `pip`:
```console
$ pip install torch-treecrf
```

## 📋 Features

- Encoding of directed graphs in an adjacency matrix, with $\mathcal{O}(1)$ retrieval of children and parents for any node, and $\mathcal{O}(N+E)$ storage.
- Support for any acyclic hierarchy representable as a [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) and not just directed trees, allowing prediction of classes such as the [Gene Ontology](https://geneontology.org).
- Multiclass output, provided all the target labels have the same number of classes: $Y \in \left\\{ 0, .., C \right\\}^L$.
- Minibatch support, with vectorized computation of the messages $\alpha_i(y_i)$ and $\beta_i(y_i)$.


## 💡 Example

To create a Tree-structured CRF, you must first define the tree encoding the
relationships between variables. Let's build a simple CRF for a root variable
with two children:

<p align="center">
  <img height="150" src="https://github.com/althonos/torch-treecrf/raw/main/static/example.svg?raw=true">
</p>

First, define an adjacency matrix $M$ representing the hierarchy, such that
$M_{i,j}$ is $1$ if $j$ is a parent of $i$:
```python
adjacency = torch.tensor([
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 0]
])
```

Then, create the a CRF with the right number of features, depending on your
feature space, like you would for a `torch.nn.Linear` module, to obtain
a Torch model:
```python
crf = torch_treecrf.TreeCRF(n_features=30, hierarchy=hierarchy)
```

If you wish to use the CRF layer only, use the `TreeCRFLayer` module,
which expects and outputs an emission tensor of shape
$(\star, C, L)$, where $\star$ is the minibatch size, $L$ the number of labels and
$C$ the number of class per label.


## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug ? Have an enhancement request ? Head over to the [GitHub issue
tracker](https://github.com/althonos/torch-treecrf/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions are more than welcome! See
[`CONTRIBUTING.md`](https://github.com/althonos/torch-treecrf/blob/main/CONTRIBUTING.md)
for more details.

## ⚖️ License

This library is provided under the [MIT License](https://choosealicense.com/licenses/mit/).

*This library was developed by [Martin Larralde](https://github.com/althonos/)
during his PhD project at the [European Molecular Biology Laboratory](https://www.embl.de/)
in the [Zeller team](https://github.com/zellerlab).*

## 📚 References

- <a id="ref1">[1]</a> Tang, Jie, Mingcai Hong, Juanzi Li, and Bangyong Liang. ‘Tree-Structured Conditional Random Fields for Semantic Annotation’. In The Semantic Web - ISWC 2006, edited by Isabel Cruz, Stefan Decker, Dean Allemang, Chris Preist, Daniel Schwabe, Peter Mika, Mike Uschold, and Lora M. Aroyo, 640–53. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer, 2006. [doi:10.1007/11926078_46](https://doi.org/10.1007/11926078_46).
- <a id="ref2">[2]</a> Pearl, Judea. ‘Reverend Bayes on Inference Engines: A Distributed Hierarchical   Approach’. In Proceedings of the Second AAAI Conference on Artificial Intelligence, 133–136. AAAI’82. Pittsburgh, Pennsylvania: AAAI Press, 1982.
- <a id="ref3">[3]</a> Bach, Francis, and Guillaume Obozinski. ‘Sum Product Algorithm and Hidden Markov Model’, ENS Course Material, 2016. http://imagine.enpc.fr/%7Eobozinsg/teaching/mva_gm/lecture_notes/lecture7.pdf.
- <a id="ref4>">[4]</a> Kschischang, Frank R., Brendan J. Frey, and Hans-Andrea Loeliger. ‘Factor Graphs and the Sum-Product Algorithm’. IEEE Transactions on Information Theory 47, no. 2 (February 2001): 498–519. [doi:10.1109/18.910572](https://doi.org/10.1109/18.910572).


