---
interact_link: content/machine-learning/02-unsupervised-learning/bayesian-methods/generative-vs-discriminative.ipynb
kernel_name: python3
has_widgets: false
title: 'Generative Vs Discriminative'
prev_page:
  url: /machine-learning/02-unsupervised-learning/bayesian-methods/README
  title: 'Bayesian Methods'
next_page:
  url: /machine-learning/02-unsupervised-learning/bayesian-methods/distributions
  title: 'Probability Distributions'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Generative VS Discriminative Machine Learning Algorithms

In this notebook, we'll go through the differences and types of generative and discriminative algorithms used in machine learning.

Note that many supervised learning algorithms are essentially estimating $P(X,Y)$ or Probability of seeing both the specific set of features and also the specific label that comes with it.

See however, that $P(X,Y)$ can be decomposed into 2 different forms based on the law of conditional probability:
$$
P(X,Y) = P(X\mid Y) \times P(Y) = \text{likelihood} \times \text{prior} \\
P(X,Y) = P(Y\mid X) \times P(X) = \text{posterior} \times \text{normalizing constant}
$$



---
# What's the Difference?

Bayes Rule: $${p(y\mid x) = \frac{p(x\mid y) * p(y)}{p(x)} }$$

- Generative
    - When we estimate $P(X,Y)=P(X\mid Y)P(Y)$ or $P(Y\mid X)P(X)$ (Not bayes rule, just law of conditional probability), then we call it generative learning. **(When we find the joint probability of features and labels)**
    - Models Likelihood = ${p(x\mid y)}$ (Probability of seeing those features given that i'm from a certain class label) and Prior = ${p(y)}$ (Probability of being the class label)
    - Creates a boundary to encompass each class like clustering
    - Makes prediction using Bayes Rule to get ${p(y=0/1\mid x)}$, ${\text{classes}=0, 1}$
        - ${p(x\mid y)}$: Model finds this
        - ${p(y)}$: Model finds this too
        - ${p(x)}$: ${p(x)} = {\sum}_{y} {p(x\mid y)} = {p(x\mid y=0)*p(y=0)} + {p(x\mid y=1)*p(y=1)}$
    - Finds parameters that explain all data.
    - Makes use of all the data.
    - Flexible framework, can incorporate many tasks (e.g. classification, regression, survival analysis, generating new data samples similar to the existing dataset, etc).
    - Stronger modeling assumptions.
    - Examples:
        - Naïve Bayes
        - Bayesian networks
        - Markov random fields
            - Used in NLP
        - Hidden Markov Models (HMM)
            - Used in NLP
- Discriminative
    - When we only estimate $P(Y\mid X)$ directly, then we call it discriminative learning. **(When we only find the posterior)**
    - Models Posterior = ${p(y\mid x)}$ directly.
    - Finds best hyperplane / boundary to separate the classes
    - Finds parameters that help to predict relevant data.
    - Learns to perform better on the given tasks.
    - Weaker modeling assumptions.
    - Less immune to overfitting.
    - Examples:
        - Logistic regression (Discriminative version of Naïve Bayes)
        - Perceptron
        - Support Vector Machine
            - Finds best hyperplane to maximize margin between nearest points from each class to hyperplane and hyperplane.
        - Traditional neural networks
        - Nearest neighbour
            - Defines similarity / distance metric and uses ${k}$ nearest neighbours to classify data point
        - Conditional Random Fields (CRF)s
            - Used in NLP



---
# Takeaways

In practice, discriminative classifiers outperform generative classifiers, if you have a lot of data.

Generative classifiers learn **P(Y\mid X)** indirectly and can get the wrong assumptions of the data distribution. 

Quoting Vapnik from Statistical Learning Theory:
one should solve the (classification) problem directly and never solve a more general problem as an intermediate step (such as modeling **P(X\mid Y)**).

A very good paper from Andrew Ng in NIPS 2001 concludes that:

a) The generative model does indeed have a higher asymptotic error (as the number of training examples become large) than the discriminative model but,

b) The generative model may also approach its asymptotic error much faster than the discriminative model – possibly with a number of training examples that is only logarithmic, rather than linear, in the number of parameters.

**So, simply said, if you have a lot of data, stick with the discriminative models.**



---
## Resources
- [Differences between what Generative and Discriminative ML algos do](https://www.youtube.com/watch?v=z5UQyCESW64)
- [Generative VS Discriminative example](https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3)
- [Kilian Weinberger's notes](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote04.html)
- [Mihaela van der Schaar's Generative VS Discriminative Notes](http://www.stats.ox.ac.uk/~flaxman/HT17_lecture5.pdf)
- [Generative and Discriminative classifiers](http://www.chioka.in/explain-to-me-generative-classifiers-vs-discriminative-classifiers/)

