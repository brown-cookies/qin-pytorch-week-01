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

## Table of Contents

- [Data](#data)
- [Softmax and Cross-Entropy Gradient Derivation](#softmax-and-cross-entropy-gradient-derivation)
- [Data from One Layer Manual Classification](#data-from-one-layer-manual-classification)
- [Error Analysis](#error-analysis)
- [Diagnosis](#diagnosis)
- [Reproducibility](#reproducibility)
  - [Quick Setup](#quick-setup)
  - [Re-running](#re-running)
- [AI Usage](#ai-usage)

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

## Softmax and Cross-Entropy Gradient Derivation

For this derivation, consider one sample with logits for the classes. The Wine
classifier has three classes.

### 1. Softmax

The softmax probability for class $j$ is:
  
$$
P_j = \frac{e^{z_j}}{\displaystyle\sum_{k=1}^{C} e^{z_k}}
$$

Let:

$$
S = \sum_{k=1}^{C} e^{z_k}
$$

Then:


$$
P_i = \frac{e^{z_i}}{S}
$$

### 2. Cross-Entropy

The cross-entropy loss for one sample is:

$$
L = -\sum_{i=1}^{C} y_i \log(P_i)
$$

Differentiate the loss with respect to $P_i$:

$$
\frac{\partial L}{\partial P_i}
= -y_i\frac{\partial}{\partial P_i}\log(P_i)
$$

Since the derivative of $\log(P_i)$ is $1/P_i$:

$$
\boxed{\frac{\partial L}{\partial P_i}=-\frac{y_i}{P_i}}
$$

### 3. Differentiate Softmax with Respect to a Logit

Starting from:

$$
P_i = \frac{e^{z_i}}{S}
$$

Apply the quotient rule with respect to $z_j$:

$$
\frac{\partial P_i}{\partial z_j}
=\frac{
S\frac{\partial e^{z_i}}{\partial z_j}
-e^{z_i}\frac{\partial S}{\partial z_j}
}{S^2}
$$

The two derivatives are:

$$
\frac{\partial e^{z_i}}{\partial z_j}=
\begin{cases}
e^{z_i} & i=j,\\
0 & i\ne j
\end{cases}
$$

and:

$$
\frac{\partial S}{\partial z_j}=e^{z_j}
$$

Now consider the two cases.

**Case A: $i=j$**

$$
\frac{\partial P_i}{\partial z_i}
=\frac{e^{z_i}S-e^{z_i}e^{z_i}}{S^2}
$$

$$
=\frac{e^{z_i}(S-e^{z_i})}{S^2}

$$

$$
=\frac{e^{z_i}}{S}\left(1-\frac{e^{z_i}}{S}\right)
$$

Since $P_i=e^{z_i}/S$:

$$
\boxed{\frac{\partial P_i}{\partial z_i}=P_i(1-P_i)}
$$

**Case B: $i\ne j$**

$$
\frac{\partial P_i}{\partial z_j}
=\frac{-e^{z_i}e^{z_j}}{S^2}
$$

$$
=-\frac{e^{z_i}}{S}\frac{e^{z_j}}{S}
$$

Since $P_i=e^{z_i}/S$ and $P_j=e^{z_j}/S$:

$$
\boxed{\frac{\partial P_i}{\partial z_j}=-P_iP_j}
$$

### 4. Apply the Chain Rule

The loss depends on the logits through the probabilities. Apply the chain
rule:

$$
\frac{\partial L}{\partial z_j}
=\sum_{i=1}^{C}
\frac{\partial L}{\partial P_i}
\frac{\partial P_i}{\partial z_j}
$$

Separate the term where $i=j$ from the terms where $i\ne j$:

$$
\frac{\partial L}{\partial z_j}
= -\frac{y_j}{P_j}P_j(1-P_j)
+\sum_{i\ne j}\left(-\frac{y_i}{P_i}\right)(-P_iP_j)
$$

Cancel the probabilities in each term:

$$
= -y_j(1-P_j)+\sum_{i\ne j}y_iP_j
$$

Expand the first term:  

$$
= -y_j+y_jP_j+\sum_{i\ne j}y_iP_j
$$

Factor out $P_j$ from the last two terms:

$$
= -y_j+P_j\left(y_j+\sum_{i\ne j}y_i\right)
$$

For one-hot encoding, the target values sum to one:

$$
\sum_{i=1}^{C}y_i=1
$$

Therefore:

$$
\frac{\partial L}{\partial z_j}=-y_j+P_j(1)
$$

The final gradient is:

$$
\boxed{\frac{\partial L}{\partial z_j}=P_j-y_j}
$$

## Data from One layer Manual Classification
```
Epoch    0 | Loss: 3.4481
Epoch   10 | Loss: 3.1378
Epoch   20 | Loss: 2.8515
Epoch   30 | Loss: 2.5900
Epoch   40 | Loss: 2.3533
Epoch   50 | Loss: 2.1406
Epoch   60 | Loss: 1.9508
Epoch   70 | Loss: 1.7823
Epoch   80 | Loss: 1.6333
Epoch   90 | Loss: 1.5019
```

## Error Analysis

## Diagnosis

## Reproducibility

### Quick Setup

### Re-running


## AI Usage
| File/Function | How it was used |
|---|---|
| `load_data()` | Asked AI for guidance on how to load, process, split training/testing of the wine dataset. Asked the concept for data preparation, StandardScaler |
| `softmax()` and `loss()` | Asked AI to explain the purpose of softmax, cross-entropy, one-hot encoded labels, and the numerical-stability steps used in the implementation. |
| `gradient()` | Used AI explanations to understand the chain rule and how the softmax and cross-entropy derivatives relate to the manual gradient. The handwritten derivation and manual gradient implementation are my own work. |