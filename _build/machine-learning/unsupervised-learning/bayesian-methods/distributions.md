---
interact_link: content/machine-learning/unsupervised-learning/bayesian-methods/distributions.ipynb
kernel_name: python3
has_widgets: false
title: 'Probability Distributions'
prev_page:
  url: /machine-learning/unsupervised-learning/bayesian-methods/generative-vs-discriminative
  title: 'Generative Vs Discriminative'
next_page:
  url: /machine-learning/unsupervised-learning/bayesian-methods/multivariate-gaussian
  title: 'Multivariate Gaussian Distribution'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Probability Distributions



---
# Discrete



## Categorical

- Special case of **Multinomial** where $n=1$
- Distribution of possible results of a random variable that can take on one of $K$ possible categories, with the probability of each category separately specified
    - e.g. If we roll a dice, what's the probability that 3 comes up?



## Bernoulli

- Special case of **Binomial** where $n=1$
- Special case of **Categorical** where $k=2$
- Distribution of the number of success in a single $n=1$ Bernoulli trial ($k=2$, success / failure)
    - e.g. If a coin is tossed once, what is the probability it comes up heads?



## Binomial

- Special case of **Multinomial** where $k=2$
- Distribution of the number of success in $n$ **i.i.d.** Bernoulli trials ($k=2$, success / failure) **with** replacement
    - e.g. If a coin is tossed 20 times, what is the probability heads comes up exactly 14 times?



## Multinomial

- Distribution of the outcome of $n$ **i.i.d.** trials, where the outcome of each trial has a categorical distribution ($k>2$, multiple classes) **with** replacement
    - e.g. If we draw 5 colored balls from a bag, what is the probability that we get 2 blue balls, 2 red balls, and 1 green ball?



## Negative Binomial

- Distribution of the number of **i.i.d.** Bernoulli trials needed to get $k$ successes
    - e.g. If a coin is repeatedly tossed, what is the probability the 3rd time heads appears occurs on the 9th toss?



## Geometric

- Special case of **Negative Binomial** where $k=1$
- Distribution of the number of **i.i.d.** Bernoulli trials needed to get the first success
    - e.g. If a coin is repeatedly tossed, what is the probability that the **first** time heads appears occurs on the 8th toss?



## Hypergeometric

- Binomial closely approximates Hypergeometric if we are sampling only a small fraction of the population
- Distribution of the number of success in $n$ **i.i.d.** Bernoulli trials ($k=2$, success / failure) **without** replacement
    - e.g. If 5 cards are drawn without replacement, what is the probability 3 hearts are drawn?



## Poisson

- Distribution of the number of **i.i.d.** events in a given time / length / area / volume
    - e.g. What is the probability there will be 4 car accidents on a university campus in a given week?



---
# Continuous



## Exponential

- Special case of **Gamma** where $n=1$, Continuous counterpart of **Geometric**
- Distribution of the time between events in a Poisson point process, i.e., a process in which events occur continuously and independently at a constant average rate.
    - e.g. How much time will pass until the next plane lands on a landing strip?



## Gamma

- Gamma : Exponential :: Binomial : Bernoulli
- Distribution of 
    - e.g. How much time will pass until $n$ planes land on any given landing strip



## Wishart / Multivariate Gamma

- Multivariate Generalization of **Gamma**
- Gamma : Wishart :: Beta : Dirichlet
- Distribution of 
    - e.g. 



## Beta

- Special case of **Dirichlet**
- Distribution over real values on the interval [0, 1]
    - e.g. What is probability of 27% of people getting infected by the disease?



## Dirichlet / Multivariate Beta

$$
f \left(x_1,\ldots, x_{K}; \alpha_1,\ldots, \alpha_K \right) = \frac{1}{\mathrm{B}(\boldsymbol\alpha)} \prod_{i=1}^K x_i^{\alpha_i - 1}
$$

where $\{x_k\}_{k=1}^{k=K}$ belong to the standard $K-1$ simplex, or in other words: $\sum_{i=1}^{K} x_i=1 \mbox{ and } x_i \ge 0 \mbox{ for all } i \in [1,K]$

The normalizing constant is the multivariate beta function, which can be expressed in terms of the gamma function:

$$
\mathrm{B}(\boldsymbol\alpha) = \frac{\prod_{i=1}^K \Gamma(\alpha_i)}{\Gamma\left(\sum_{i=1}^K \alpha_i\right)},\qquad\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_K)
$$

- Multivariate Generalization of **Beta**
- Beta : Dirichlet :: Binomial : Multinomial :: Gamma : Wishart
- Conjugate Prior of Multinomial and Categorical
- When $\alpha=1$, Dirichlet is essentially a uniform distribution as it gives each $x_i$ equally probability of $\frac{1}{\text{normalizing constant}\,{\mathrm{B}(\boldsymbol\alpha)} }$, this is called a flat Dirichlet distribution
- Distribution over vectors whose values are all in the interval [0, 1] and sum of values in the vector = 1, AKA a probability simplex
    - e.g. 



## Symmetric Dirichlet

- Special case of **Dirichlet** where vector $\mathbf{\alpha}$ has all the same values



---
## Conjugate Priors

- If the posterior distributions $p(\theta \mid X)$ are in the same probability distribution family as the prior probability distribution $p(\theta)$, the prior and posterior are then called conjugate distributions, and the prior is called a conjugate prior for the likelihood function $p(X \mid \theta)$.

Probability distribution families:
1. Exponential Family
    - normal
    - exponential
    - gamma
    - chi-squared
    - beta
    - Dirichlet
    - Bernoulli
    - categorical
    - Poisson
    - Wishart
    - inverse Wishart
    - geometric
    - binomial (with fixed number of trials)
    - multinomial (with fixed number of trials)
    - negative binomial (with fixed number of failures)



---
## Resources
- [Overview of Some Discrete Probability Distributions(Binomial,Geometric,Hypergeometric,Poisson,NegB)](https://www.youtube.com/watch?v=UrOXRvG9oYE)
- [Conjugate Prior Wiki](https://en.wikipedia.org/wiki/Conjugate_prior)
- [Exponential family wiki](https://en.wikipedia.org/wiki/Exponential_family)
- [What is the gamma distribution used for?](https://www.quora.com/What-is-gamma-distribution-used-for)
- [Beta & Dirichlet distribution video](https://www.youtube.com/watch?v=CEVELIz4WXM)

