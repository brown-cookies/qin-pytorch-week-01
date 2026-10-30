# A neural network from first principles

## Problem

A colleague has asked for a small classifier on a real table of data, and they
have one concern: they do not trust a result they cannot explain. Frameworks
hide a lot, and "it trains" is not evidence that it is right. So they want to
see the machinery.

You are to build a small neural network classifier in which the gradients are
computed by you, by hand, rather than by the framework. Then you prove those
gradients are correct by checking them against autograd. Then you rebuild the
same classifier the idiomatic way, with `nn.Module` and an optimiser, and show
that the two versions agree with each other. Finally, you package it, test it
and train it on a real dataset, so that someone else can pick it up, run it and
get your numbers.

What matters is that you can say what the abstraction is hiding, and that you
can show it rather than assert it.

## Data

The dataset is the Wine recognition dataset, loaded with
`sklearn.datasets.load_wine` from scikit-learn. It is bundled with the library,
so no download or login is needed. scikit-learn's copy comes from the UCI
Machine Learning Repository
([scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)).

| | |
|---|---|
| Rows | 178 |
| Features | 13, all numeric (chemical measurements such as alcohol, proline, flavanoids) |
| Classes | 3 (`class_0`: 59 rows, `class_1`: 71 rows, `class_2`: 48 rows) |
| Missing values | None |

## About the `notebook/` and `src/`

The `notebook/` is used for experimentation, exploration, and learning.
The `src/` is where the actual code implementation, after `notebook/`.

## AI Usage
| | |
|---|---|
| File/Function | How it was used |
| `load_data()` | Asked AI for guidance on how to load, process, split training/testing of the wine dataset. Asked the concept for data preparation, StandardScaler |