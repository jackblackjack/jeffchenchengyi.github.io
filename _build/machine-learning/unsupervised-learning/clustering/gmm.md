---
interact_link: content/machine-learning/unsupervised-learning/clustering/gmm.ipynb
kernel_name: python3
has_widgets: false
title: 'Gaussian Mixture Models'
prev_page:
  url: /machine-learning/unsupervised-learning/clustering/k-means
  title: 'K-Means'
next_page:
  url: /machine-learning/unsupervised-learning/bayesian-methods/bayesian-methods
  title: 'Bayesian Methods'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Gaussian Mixture Models



Let 

$$
    \mathbf{x} 
    = (\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_n) 
    = (\vec{x}_{1}, \vec{x}_{2}, \ldots, \vec{x}_{n}) 
    = (\begin{bmatrix}
        \text{feat }1\text{ of }\vec{x}_{1}\\
        \text{feat }2\text{ of }\vec{x}_{1}\\
        \vdots\\
        \text{feat }d\text{ of }\vec{x}_{1}
    \end{bmatrix},
    \begin{bmatrix}\text{feat }1\text{ of }\vec{x}_{2}\\
        \text{feat }2\text{ of }\vec{x}_{2}\\
        \vdots\\
        \text{feat }d\text{ of }\vec{x}_{2}
    \end{bmatrix}, 
    \ldots,
    \begin{bmatrix}
        \text{feat }1\text{ of }\vec{x}_{n}\\
        \text{feat }2\text{ of }\vec{x}_{n}\\
        \vdots\\
        \text{feat }d\text{ of }\vec{x}_{n}
    \end{bmatrix})
$$ 

be a sample of $n$ independent observations from a mixture model of two multivariate normal distributions of dimension $d$, and let 

$$\mathbf{z} = (z_1,z_2,\ldots,z_n)$$ 

be the unobserved latent variables that determine the component from which the observation originates. In the context of GMM, if we believe our data to contain 3 clusters,

$$
z_i = 
\begin{cases}
    k = \text{Cluster 1}\\    
    k = \text{Cluster 2}\\
    k = \text{Cluster 3}
\end{cases}, k = 1, \ldots, K
$$

In the context of GMM, we believe that each cluster will be sampled from a Gaussian distribution,

$$
P(\mathbf{x}_i \mid z_i = \text{Cluster 1}) \sim \mathcal{N}_d(\boldsymbol{\mu}_1,\Sigma_1)\\
P(\mathbf{x}_i \mid z_i = \text{Cluster 2}) \sim \mathcal{N}_d(\boldsymbol{\mu}_2,\Sigma_2)\\
P(\mathbf{x}_i \mid z_i = \text{Cluster 3}) \sim \mathcal{N}_d(\boldsymbol{\mu}_3,\Sigma_3)
$$



---
## MLE
**Goal**: Find $\operatorname*{argmax}_{\theta}\,P(Data;\theta)$

### 1. Incomplete-data Log-Likelihood

If our observations $X_i$ come from a mixture model with $K$ mixture components, the marginal probability distribution of $X_i$ is of the form: 
$$ P(X_i = x) = \sum_{k=1}^K \pi_kP(X_i=x|Z_i=k)$$ 
where $Z_i \in \{1,\ldots,K\}$ is the latent variable representing the mixture component for $X_i$, $P(X_i|Z_i)$ is the **mixture component**, and $\pi_k$ is the **mixture proportion** representing the probability that $X_i$ belongs to the $k$-th mixture component, and $\sum_{k=1}^K \pi_k = 1$.

$N(\mu, \Sigma)$ denote the probability distribution function for a normal random variable. In this scenario, we have that the conditional distribution $X_i|Z_i = k \sim N(\mu_k, \Sigma_k)$ so that the marginal distribution of $X_i$ is: 
$$ 
P(X_i = x) = \sum_{k=1}^K P(Z_i = k) P(X_i=x | Z_i = k) = \sum_{k=1}^K \pi_k N(x; \mu_k, \Sigma_k)
$$

Similarly, the joint probability of observations $X_1,\ldots,X_n$ is therefore: 
$$\prod_{i=1}^n \sum_{k=1}^K \pi_k N(x_i; \mu_k, \Sigma_k)$$

This note describes the EM algorithm which aims to obtain the maximum likelihood estimates of $\pi_k, \mu_k$ and $\Sigma_k$ given a data set of observations $\{x_1,\ldots, x_n\}$.

#### Computing MLE estimate for Incomplete-data Log-Likelihood
$$
\begin{aligned}
    \hat{\theta}_{MLE} &= \operatorname*{argmax}_{\theta} \,P(Data; \theta) \\
    &= \operatorname*{argmax}_{\theta} \,P(X_1=x_1,\ldots,X_n=x_n; \theta) \\
    &= \operatorname*{argmax}_{\theta} \,\prod_{i=1}^n P(X_i=x;\theta) \because \text{ Each sample is i.i.d. } \\
    &= \operatorname*{argmax}_{\theta} \,\prod_{i=1}^n \sum_{k=1}^K P(X_i=x, Z_i = k;\theta) \because \text{ Marginalization } \\
    &= \operatorname*{argmax}_{\theta} \,\prod_{i=1}^n \sum_{k=1}^K P(Z_i = k) P(X_i=x | Z_i = k;\theta) \because P(A, B) = P(A \mid B) \times P(B)\\
    &= \operatorname*{argmax}_{\theta} \,\prod_{i=1}^n \sum_{k=1}^K \pi_k N(x_i; \mu_k, \Sigma_k), \Sigma_k \succ 0 \\
    &= \operatorname*{argmax}_{\theta} \,\sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k N(x_i;\mu_k, \Sigma_k) \right ) \because \href{https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood}{ \text{logarithms are strictly increasing} }
\end{aligned}
$$
Positive semi-definite to be valid covariance matrices, or we can relax this constraint by saying that the covariance matrices must be diagonal, meaning that the entries on the diagonal are > 0, while everything else in the matrix is 0 - this corresponds to having 0 covariances meaning that our ellipsoids will be aligned with the axis and not rotated. Now we can then train our model using SGD...

$\hat{\mu_k}_{MLE}$: 
$$
\begin{aligned}
    \frac{\partial}{\partial \mu_k}\,\sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k N(x_i;\mu_k, \Sigma_k) \right ) &= 0 \\
    \sum_{i=1}^n \frac{\partial}{\partial \mu_k}\,\log \left( \sum_{k=1}^K \pi_k N(x_i;\mu_k, \Sigma_k) \right ) &= 0 \\
    \sum_{i=1}^n \frac{1}{\sum_{k=1}^K \pi_k N(x_i;\mu_k, \Sigma_k)} \sum_{k=1}^K\pi_k \frac{\partial}{\partial \mu_k}\, N(x_i;\mu_k, \Sigma_k) &= 0 \\ 
    \sum_{i=1}^n \frac{1}{\sum_{k=1}^K \pi_k N(x_i;\mu_k, \Sigma_k)} \sum_{k=1}^K\pi_k \sum_{i=1}^n { {\Sigma}^{-1}(x_i - \mu_k)} &= 0 \\
    \vdots \\
    \text{Stuck because our parameters are coupled...}
\end{aligned}
$$

### 2. Complete-data Log Likelihood

Now instead if we already know what the distribution of $Z$ is, we can find:

$$
\begin{aligned}
    P(X, Z;\theta = \mu, \Sigma, \pi) &= \prod_{i=1}^n \prod_{k=1}^K \pi_k^{I(Z_i = k)} N(x_i|\mu_k, \Sigma_k)^{I(Z_i = k)} \\
    &= \sum_{i=1}^n \sum_{k=1}^K I(Z_i = k)\left( \log (\pi_k) + \log (N(x_i|\mu_k, \Sigma_k) )\right) \because \text{ Logarithms are strictly increasing}
\end{aligned}
$$

, where 
$
I(Z_i = k) 
\begin{cases} 
    1 \text{ if } Z_i = k \\ 
    0 \text{ if } Z_i \neq k 
\end{cases}
$

## Expectation Maximization Algorithm

To use our complete data log likelihood to calculate MLE estimates:

$$
\begin{aligned}
    \hat{\theta}_{MLE} &= \operatorname*{argmax}_{\theta} \,P(\mathcal{Data}=X;\theta) \\
    &\approx \operatorname*{argmax}_{\theta} \,Q(\theta^{(t)}, \theta^{(t-1)}) \\
    &= \operatorname*{argmax}_{\theta} \,E_{Z|X,\theta^{t-1} }\left [\log (P(X,Z;\theta^{(t)})) \right] \\
    &= \operatorname*{argmax}_{\theta} \,\sum_{i=1}^n \sum_{k=1}^K E_{Z\mid X,\theta^{t-1} }\left [I(Z_i = k)\right]\left( \log ({\pi_k}^{(t)}) + \log (N(x_i;{\mu_k}^{(t)}, {\Sigma_k}^{(t)}) )\right) \\
    &= \operatorname*{argmax}_{\theta} \,\sum_{i=1}^n \sum_{k=1}^K P(Z_i=k \mid X;\theta^{t-1}) \left( \log ({\pi_k}^{(t)}) + \log (N(x_i|{\mu_k}^{(t)}, {\Sigma_k}^{(t)}) )\right) \because E_{Z \mid X}[I(Z_i = k)] = P(Z_i=k \mid X) \\
    &= \operatorname*{argmax}_{\theta} \,\sum_{i=1}^n \sum_{k=1}^K \frac{P(X_i \mid Z_i = k; \theta^{(t-1)}) \times P(Z_i = k;\theta^{(t-1)})}{P(X_i;\theta^{(t-1)})} \left( \log ({\pi_k}^{(t)}) + \log (N(x_i|{\mu_k}^{(t)}, {\Sigma_k}^{(t)}) )\right) \\
    &= \operatorname*{argmax}_{\theta} \,\sum_{i=1}^n \sum_{k=1}^K \frac{N(x_i|{\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}) \times {\pi_k}^{(t-1)})}{\sum_{k=1}^K N(x_i|{\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)})\times {\pi_k}^{(t-1)} } \left( \log ({\pi_k}^{(t)}) + \log (N(x_i|{\mu_k}^{(t)}, {\Sigma_k}^{(t)}) )\right)
\end{aligned}
$$

### Step 1: Expectation Step
- Choose initial values for ${\mu_k}^{(t-1=0)}, {\Sigma_k}^{(t-1=0)}, {\pi_k}^{(t-1=0)}$ and use these in the E-step to evaluate the **Responsibilities**: $P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1=0)}, {\Sigma_k}^{(t-1=0)}, {\pi_k}^{(t-1=0)})$

### Step 2: Maximization Step
- With $P(Z_i=k |X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)})$ fixed, maximize the expected complete log-likelihood above with respect to ${\mu_k}^{(t)}, {\Sigma_k}^{(t)}, {\pi_k}^{(t)}$

$\hat{ {\pi_k}^{(t)}_{MLE} }$:
$$
\begin{aligned}
    \frac{\partial}{\partial {\pi_k}^{(t)} } \sum_{i=1}^n \sum_{k=1}^K P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)}) \left( \log ({\pi_k}^{(t)}) + \log (N(x_i\mid{\mu_k}^{(t)}, {\Sigma_k}^{(t)})\right) &= 0 \\
    \sum_{i=1}^n \sum_{k=1}^K P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)}) \frac{\partial}{\partial {\pi_k}^{(t)} } \log ({\pi_k}^{(t)}) &= 0 \\
    \sum_{i=1}^n \sum_{k=1}^K \frac{P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)})}{ {\pi_k}^{(t)} } &= 0 \\
    \hat{ {\pi_k}^{(t)}_{MLE} } &= \frac{1}{n} \sum^{n}_{i=1} P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)}) \\
    &= \frac{1}{n} \sum^{n}_{i=1} \frac{N(x_i|{\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}) \times {\pi_k}^{(t-1)})}{\sum_{k=1}^K N(x_i|{\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)})\times {\pi_k}^{(t-1)} }
\end{aligned}
$$

$\hat{ {\mu_k}^{(t)}_{MLE} }$:
$$
\begin{aligned}
    \hat{ {\mu_k}^{(t)}_{MLE} } &= \frac{\sum^{n}_{i=1} P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)}) \times x_i}{\sum^{n}_{i=1} P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)})} \\
\end{aligned}
$$

$\hat{ {\Sigma_k}^{(t)}_{MLE} }$:
$$
\begin{aligned}
    \hat{ {\mu_k}^{(t)}_{MLE} } &= \frac{\sum^{n}_{i=1} P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)}) \times (x_i - {\mu_i}^{(t)}){(x_i - {\mu_i}^{(t)})}^\top}{\sum^{n}_{i=1} P(Z_i=k \mid X; \theta^{(t-1)}={\mu_k}^{(t-1)}, {\Sigma_k}^{(t-1)}, {\pi_k}^{(t-1)})} \\
\end{aligned}
$$

### Step 3: Evaluate Log Likelihood

Evaluate:
$$\sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k N(x_i;\mu_k, \Sigma_k) \right )$$

If there is no convergence, go back to Step 2: Expectation Step



---
# Picking K

- Occam's Razor: Pick "simplest" of all models that fit
    - Bayes Information Criterion (BIC): $\operatorname*{max}_{p} \{L - \frac{1}{2} \log n\}$
    - Akaike Information Criterion (AIC): $\operatorname*{min}_{p} \{2p - L\}$
        - $L$ - Likelihood, how well model fits data
        - $p$ - Number of parameters, how "simple/generalizable" model is



---
# How to turn GMM into Kmeans

- Peg variance = 1
- Assume uniform priors



---
## Resources:
- [Quick qwalkthrough of EM in GMM](https://mas-dse.github.io/DSE210/Additional%20Materials/gmm.pdf)
- [Matthew Stephen's Blog on Mixture Models](https://stephens999.github.io/fiveMinuteStats/intro_to_mixture_models.html)
- [Matthew Stephen's Blog on GMM-EM](https://stephens999.github.io/fiveMinuteStats/intro_to_em.html)
- [Valentin Wolf's GMM implementation](https://github.com/volflow/Expectation-Maximization/blob/master/9.2%20Mixtures%20of%20Gaussians%20and%20Expectation-Maximization.ipynb)
- [Incomplete VS Complete log likelihood in GMMs](https://www.cs.utah.edu/~piyush/teaching/gmm.pdf)
- [GMM-EM Wiki](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture)
- [Deriving MLE estimates for Gaussian Distribution Parameters](https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian)
- [Indian Institute of Technology notes](http://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf)

