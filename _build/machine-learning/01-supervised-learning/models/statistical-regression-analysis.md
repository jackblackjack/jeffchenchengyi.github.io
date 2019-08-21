---
interact_link: content/machine-learning/01-supervised-learning/models/statistical-regression-analysis.ipynb
kernel_name: python3
has_widgets: false
title: 'Statistical Regression Analysis'
prev_page:
  url: /machine-learning/01-supervised-learning/models/logistic-regression
  title: 'Logistic Regression'
next_page:
  url: /machine-learning/01-supervised-learning/models/multilabel-classification
  title: 'Multi-Label models'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Statistical Regression Analysis

We'll walkthrough the modeling phase of statistical regression analysis in this notebook and also the bulk of the basics of econometrics.

### Table of Contents
1. [Simple Linear Regression](#simplelinreg)
2. [Multiple Linear Regression](#multiplelinreg)
3. [General Linear Models](#glm)
4. [Generalized Linear Models](#glim)

## Preliminaries

### Gauss-Markov Assumptions<a id='gauss-markov'></a>
1. Linear in Parameters
    - Good: $y_i = w_0 + w_1{(x_1)}_i + \epsilon_i$
    - Good: $y_i = w_0 + w_1{(x_1)}^2_i + \epsilon_i$
    - Bad: $y_i = w_0w_1{(x_1)}_i + \epsilon_i$ 
2. $(\mathbf{x}_i = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_m \end{bmatrix}, y_i)$ are a random sample and come from the same population / same distribution
3. Zero Conditional Mean of Error $\mathbb{E}[\epsilon_i \vert \mathbf{x}_i] = 0$
4. No Perfect Collinearity / Column Vectors in Design Matrix $X$ are linearly independent
    - There are no features that are linear functions of each other
5. Homoscedastic Errors
    - $Var(\epsilon_i) = \sigma^2$ and $Var(\epsilon_i \vert \mathbf{x}_i) = \sigma^2$
6. No Serial Correlation
    - $Cov(\epsilon_i, \epsilon_j) = 0$, meaning that knowing the error on one sample does not help to predict another's (independent portion of i.i.d.)



---
# Simple Linear Regression<a id='simplelinreg'></a>

### Ground Truth Model

#### One Sample
$$
y_i = w_0 + w_1{(x_1)}_i + \epsilon_i,\,i \in [1, N]\,\text{(Index of Sample)}\,,\epsilon_i \vert {(x_1)}_i \sim \mathcal{N}(\mu=0, \sigma^2)
$$
Errors $\epsilon_i$ are assumed to be identically, independently, and normally distributed with a mean of 0 given a sample ${x_1}_i$ (More on this @ [Gauss-Markov Assumptions](#gauss-markov))

$$
\begin{aligned}
\mathbb{E}[y_i \vert {(x_1)}_i] &= \mathbb{E}[w_0 + w_1{(x_1)}_i + \epsilon_i \vert {(x_1)}_i] \\
&= \mathbb{E}[w_0 \vert {(x_1)}_i] + \mathbb{E}[w_1{(x_1)}_i \vert {(x_1)}_i] + \mathbb{E}[\epsilon_i \vert {(x_1)}_i] \\
&= w_0 + w_1\mathbb{E}[{(x_1)}_i \vert {(x_1)}_i] + 0 \\
&= w_0 + w_1{(x_1)}_i \\
\end{aligned}
$$



#### All Samples (Vectorized)
$$
\begin{aligned}
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N \\
\end{bmatrix} &= 
\begin{bmatrix}
w_0 + w_1{(x_1)}_1 + \epsilon_1 \\
w_0 + w_1{(x_1)}_2 + \epsilon_2 \\
\vdots \\
w_0 + w_1{(x_1)}_N + \epsilon_N \\
\end{bmatrix} \\
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N \\
\end{bmatrix} &= 
w_0
\begin{bmatrix}
1 \\
1 \\
\vdots \\
1 \\
\end{bmatrix} +
w_1
\begin{bmatrix}
{(x_1)}_1 \\
{(x_1)}_2 \\
\vdots \\
{(x_1)}_N \\
\end{bmatrix} + 
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_N \\
\end{bmatrix} \\
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N \\
\end{bmatrix} &= 
\begin{bmatrix}
1 & {(x_1)}_1 \\
1 & {(x_1)}_2 \\
\vdots & \vdots \\
1 & {(x_1)}_N \\
\end{bmatrix}
\begin{bmatrix}
w_0 \\
w_1 \\
\end{bmatrix} + 
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_N \\
\end{bmatrix} \\
\underset{N \times 1}{\mathbf{y} } &= \underset{N \times m}{X}\,\underset{m \times 1}{\mathbf{w} } + \underset{N \times 1}{\mathbf{\epsilon} }
\end{aligned}
$$ $X$ is the [*Design Matrix*](https://en.wikipedia.org/wiki/Design_matrix)



### Least Squares Estimator
$$
\hat{y_i} = \hat{w_0} + \hat{w_1}{(x_1)}_i
$$

How do we get $\hat{w_0}$ ($y$-intercept) and $\hat{w_1}$ (gradient of slope)?

Sum of Squares (SS):
$$
\begin{aligned}
SS &= \sum^N_{i=1}{(y_i - \hat{y_i})}^2 \\
&= \sum^N_{i=1}{(y_i - \hat{w_0} - \hat{w_1}{(x_1)}_i)}^2 \\
\end{aligned}
$$

First Order Conditions:
$$
\begin{aligned}
(I): \frac{\partial SS}{\partial w_0} = -2\sum^N_{i=1}{(y_i - \hat{w_0} - \hat{w_1}{(x_1)}_i)} &= 0 \\
\sum^N_{i=1}y_i - \sum^N_{i=1}\hat{w_0} - \sum^N_{i=1}\hat{w_1}{(x_1)}_i) &= 0 \\
\sum^N_{i=1}y_i &= \sum^N_{i=1}\hat{w_0} + \sum^N_{i=1}\hat{w_1}{(x_1)}_i  \\
N\bar{y} &= \hat{w_0}N + \hat{w_1}N\bar{x_1}  \\
\bar{y} &= \hat{w_0} + \hat{w_1}\bar{x_1}  \\
\hat{w_0} &= \bar{y} - \hat{w_1}\bar{x_1} \\
(II): \frac{\partial SS}{\partial w_1} = -2\sum^N_{i=1}{(x_1)}_i{(y_i - \hat{w_0} - \hat{w_1}{(x_1)}_i)} &= 0 \\
\sum^N_{i=1}{(x_1)}_iy_i - \sum^N_{i=1}\hat{w_0}{(x_1)}_i - \sum^N_{i=1}\hat{w_1}{ {(x_1)}_i}^2 &= 0 \\
\sum^N_{i=1}{(x_1)}_iy_i &= \sum^N_{i=1}\hat{w_0}{(x_1)}_i + \sum^N_{i=1}\hat{w_1}{ {(x_1)}_i}^2 \\
\sum^N_{i=1}{(x_1)}_iy_i &= \hat{w_0}N\bar{x_1} + \hat{w_1}\sum^N_{i=1}{ {(x_1)}_i}^2 \\
\sum^N_{i=1}{(x_1)}_iy_i &= (\bar{y} - \hat{w_1}\bar{x_1})N\bar{x_1} + \hat{w_1}\sum^N_{i=1}{ {(x_1)}_i}^2 \,\because\,(I)\\
\sum^N_{i=1}{(x_1)}_iy_i - N\bar{x_1}\bar{y} &= \hat{w_1}(N\bar{x_1}\bar{x_1} + \sum^N_{i=1}{ {(x_1)}_i}^2) \\
\hat{w_1} &= \frac{\sum^N_{i=1}{(x_1)}_iy_i - N\bar{x_1}\bar{y} }{N\bar{x_1}^2 + \sum^N_{i=1}{ {(x_1)}_i}^2} \\
&= \frac{\sum^N_{i=1}({(x_1)}_iy_i - \bar{x_1}y_i)}{\sum^N_{i=1}({ {(x_1)}_i}^2 - \bar{x_1}{(x_1)}_i)} \\
&= \frac{\sum^N_{i=1}y_i({(x_1)}_i - \bar{x_1})}{\sum^N_{i=1}({ {(x_1)}_i}^2 - \bar{x_1}{(x_1)}_i)} \\
&= \frac{\sum^N_{i=1}y_i({(x_1)}_i - \bar{x_1})}{\sum^N_{i=1}{(x_1)}_i({ {(x_1)}_i} - \bar{x_1})} \\
&= \frac{\sum^N_{i=1}({(x_1)}_i - \bar{x})(y_i - \bar{y})}{\sum^N_{i=1}{({(x_1)}_i - \bar{x})}^2} \\
&= \frac{COV({(x_1)}_i, y_i)}{Var({(x_1)}_i)}
\end{aligned}
$$

Note: From $(I)$, we know that our line of best fit will definitely have to pass through the mean of all target variables $\bar{y}$.

Hence, our Least Squares Estimates are:
$$
\begin{aligned}
\hat{w_0} &= \bar{y} - \hat{w_1}\bar{x_1} \\
\hat{w_1} &= \frac{Cov({(x_1)}_i, y_i)}{Var({(x_1)}_i)}
\end{aligned}
$$



---
# Polynomial Regression

$$
y = w_0 + w_1x + w_2x^2 + w_3x^3 + w_4x^4 + \ldots + w_nx^n
$$



---
# Multiple Linear Regression<a id='multiplelinreg'></a>

$$
y = w_0 + w_1x_1 + w_2x_2 +w_3x_3 + \ldots + w_nx_n
$$



---
# General Linear Models<a id='glm'></a>

### Anova
--- Refer to [Statistics Review]() ---

### Ancova

### Multivariate Linear Regression



---
# Generalized Linear Models<a id='glim'></a>

### Poisson Regression
- Used to model Count data
- We assume the target variable we're trying to model to be a random variable with Poisson Distribution

$$
\begin{aligned}
P(y_i = k) &= \frac{e^{-\mu_i} \mu_i^k}{k!},\,y_i \in \mathbb{Z}^+,\,\mu_i = e^{\mathbf{w}^\top x_i} \\
P(y; X\mathbf{w})&= \prod^{n}_{i=1} \frac{(e)^{(-{e)^{(\mathbf{w}^\top x_i)}}} {(e)^{(\mathbf{w}^\top x_i)}}^{(y_i)}}{y_i!} \\
\end{aligned}
$$

Negative Log-Likelihood: 
$$
\begin{aligned}
-log(P(y; X\mathbf{w})) &= -log(\prod^{n}_{i=1} \frac{(e)^{(-{e)^{(\mathbf{w}^\top x_i)}}} {(e)^{(\mathbf{w}^\top x_i)}}^{(y_i)}}{y_i!}) \\
&= -\sum^{n}_{i=1} (-{e^{\mathbf{w}^\top x_i}} + {y_i \mathbf{w}^\top x_i} - log(y_i!)) \\
&= \sum^{n}_{i=1} ({e^{\mathbf{w}^\top x_i}} - {y_i \mathbf{w}^\top x_i} 
+ log(y_i!)) \\
\end{aligned}
$$

Convex Optimization Objective:
$$
\begin{aligned}
\underset{\mathbf{w}}{\text{minimize}}\,\sum^{n}_{i=1} ({e^{\mathbf{w}^\top x_i}} - {y_i \mathbf{w}^\top x_i})
\end{aligned}
$$

### Logistic Regression and Probit Regression
- Used for Binary data

### Multinomial Logistic Regression and Multinomial Probit Regression
- Used for Categorical data

### Ordered Logit and Ordered Probit Regression
- Used for Ordinal data



---
## Resources:
- [Ben Lambert's Full course of Undergrad Econometrics](https://www.youtube.com/playlist?list=PLwJRxp3blEvZyQBTTOMFRP_TDaSdly3gU)
- [Casualty Actuarial Society Forum Spring 2013](https://www.casact.org/pubs/forum/13spforum/Semenovich.pdf)

