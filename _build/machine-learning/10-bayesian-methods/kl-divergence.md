---
interact_link: content/machine-learning/10-bayesian-methods/kl-divergence.ipynb
kernel_name: python3
has_widgets: false
title: 'Kullback-Liebler Divergence'
prev_page:
  url: /machine-learning/10-bayesian-methods/generative-vs-discriminative
  title: 'Generative Vs Discriminative'
next_page:
  url: /machine-learning/10-bayesian-methods/pgm
  title: 'Probabilistic Graphical Models'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Kullback-Liebler (KL) Divergence

Important Concept for understanding Variational Auto-Encoders!


### Jensen's Inequality
If $X$ is a random variable and $f$ is a concave function (if function is concave, any line segment between two points will lie below the function):

$$
    f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
$$

If $X$ is a random variable and $f$ is a convex function (if function is convex, any line segment between two points will lie above the function):

$$
    f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
$$

### Entropy

Entropy = Amount of Uncertainty, $\therefore$ random events with uniform probability have the highest entropy

Case 1: 
- We have 2 events, 
- uniform probability $\therefore$ each happening with $p = 0.5$
- Transmitting 1 bit of information = Reduce uncertainty by 50%
- Minimum number of bits required: 
    - 1 bit

Case 2: 
- We have 8 events, 
- uniform probability $\therefore$ each happening with $p = \frac{1}{8}$
- Minimum number of bits required: 

$$3 \text{ bits} = {log}_{2} (8 \text{ events}) = -{log}_{2} (p = \frac{1}{8})$$

Case 3: 
- We have 2 events, 
- event 1 happens with $p_1 = 0.75$, event 2 happens with $p_2 = 0.25$
- Average / Minimum number of bits required: 

$$
\begin{aligned}
&(p_1 = 0.75) \times -{log}_{2} (p_1 = 0.75) + (p_1 = 0.25) \times -{log}_{2} (p_1 = 0.25) \\
&= (p_1 = 0.75) \times 0.41 + (p_1 = 0.25) \times 2 \\
&= 0.81 \text{ bits}
\end{aligned}
$$
    
$$\therefore \text{Entropy: } H(p) = -\sum_{i=1}^n {\mathrm{p}(x_i) \log_b \mathrm{p}(x_i)}\,\{\text{for discrete }x\} \\ = -\int_{x} {\mathrm{p}(x) \log_b \mathrm{p}(x)}{dx}\,\{\text{for continuous }x\}$$
- $[ {b = 2:}\text{ bits},  {b = e:}\text{ nats}, {b = 10:}\text{ bans} ]$

### Cross-entropy
Case 4:
- We have 8 events,

|Events|Actual True probability distribution of events $p$|Predicted probability distribution $q$|
|-|-|-|
|1|0.01|0.25|
|2|0.01|0.25|
|3|0.04|0.125|
|4|0.04|0.125|
|5|0.10|0.0625|
|6|0.10|0.0625|
|7|0.35|0.03125|
|8|0.35|0.03125|

- To find out how many bits that were sent over were actually useful, we sum product the number of bits sent over according to our predicted distribution with the actual probabilities of each event:

$$\therefore \text{Cross-Entropy: } H(p, q) = -\sum_{i=1}^n {\mathrm{p}(x_i) \log_b \mathrm{q}(x_i)}\,\{\text{for discrete }x\} \\ = -\int_{x} {\mathrm{p}(x) \log_b \mathrm{q}(x)}{dx}\,\{\text{for continuous }x\}$$

- Asymetric: $H(p, q) \neq H(q, p)$

### KL Divergence

The KL divergence tells us how well the probability distribution Q approximates the probability distribution P by calculating the cross-entropy minus the entropy.

$$
\therefore \text{KL Divergence}\,=\,\text{Cross Entropy}-\text{Entropy} \\
{D}_{KL}(p \parallel q) = H(p, q) - H(p) \\
{D}_{KL}(p \parallel q) = \sum_{i=1}^n \mathrm{p}(x_i) \log_b \frac{ {\mathrm{p}(x_i)} }{\mathrm{q}(x_i)}\,\{\text{for discrete }x\} \\
{D}_{KL}(p \parallel q) = \int_{x} {\mathrm{p}(x) \log_b \frac{\mathrm{p}(x)}{\mathrm{q}(x)} }{dx}\,\{\text{for continuous }x\} \\
{D}_{KL}(p \parallel q) = \mathbb{E}_p \left(\log\frac{p}{q}\right)
$$

- Non-negative: 
$${D}_{KL}(p \parallel q) \geq 0 \because -{D}_{KL}(p \parallel q) = \mathbb{E}_p \left(-\log \frac{p}{q}\right) = \mathbb{E}_p \left(\log \frac{q}{p}\right) \leq \log\left(\mathbb{E}_p \frac{q}{p}\right) = \log\int p(x) \frac{q(x)}{p(x)} dx = \log(1) = 0$$
- Asymmetric: 
$${D}_{KL}(p \parallel q) \neq {D}_{KL}(q \parallel p)$$
- $${D}_{KL}(p \parallel p) = 0$$
- You can think of KL Divergence as a mean of the difference in probabilities at each point $x_i$ in the log scale.



---
## Resources:
- [Naoki Shibuya's article on Entropy](https://towardsdatascience.com/demystifying-entropy-f2c3221e2550)
- [Naoki Shibuya's article on Cross Entropy](https://towardsdatascience.com/demystifying-cross-entropy-e80e3ad54a8)
- [Naoki Shibuya's article on KL Divergence](https://towardsdatascience.com/demystifying-kl-divergence-7ebe4317ee68)
- [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)
- [Entropy (Information) Wiki](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Space Worms and KL Divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
- [Information Entropy by Khan Academy](https://www.youtube.com/watch?v=2s3aJfRr9gE)

