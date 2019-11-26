---
interact_link: content/machine-learning/08-genetic-algorithms/overview-of-es.ipynb
kernel_name: python3
has_widgets: false
title: 'Overview of Evolutionary Strategies'
prev_page:
  url: /machine-learning/08-genetic-algorithms/saga-fpga.html
  title: 'Evolutionary Algorithms on FPGAs'
next_page:
  url: /machine-learning/08-genetic-algorithms/evolutionary-strategies.html
  title: 'Evolutionary Strategies'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Evolution Strategies

### Table of Contents
1. [Simple Evolutionary Strategy](#ses)
2. [Simple Genetic Algorithm](#sga)
3. [Covariance-Matrix Adaptation Evolutionary Strategy (CMA-ES)](#cma-es)
4. [Natural Evolution Strategy](#nes)
5. [OpenAI Evolution Strategy](#oais)



---
## Simple Evolutionary Strategy<a id='ses'></a>



1. Initialize $\mu = (\mu_x, \mu_y) = (0, 0)$ and **fix** $\sigma = (\sigma_x, \sigma_y)$
2. Sample $N$ coordinate pairs $(x, y)$ from multivariate gaussian with $\mu$ and $\sigma$
3. For each $(x, y)$ coordinate pair sampled, evaluate the function (e.g. Schaffer or Rastrigin - these were used because of many local optima) and obtain a fitness score
4. Set new mean $\mu = (x_{fittest}, y_{fittest})$ and repeat until convergence.

Advantages:
- Simple

Disadvantages:
- Get stuck in local optima, especially when $\sigma$ is chosen badly (in my opinion)



---
## Simple Genetic Algorithm<a id='sga'></a>



1. Initialize $\mu = (\mu_x, \mu_y) = (0, 0)$ and **fix** $\sigma = (\sigma_x, \sigma_y)$ [Same as simple ES]
2. Sample $N$ coordinate pairs $(x, y)$ from multivariate gaussian with $\mu$ and $\sigma$ [Same as simple ES]
3. For each $(x, y)$ coordinate pair sampled, evaluate the function (e.g. Schaffer or Rastrigin - these were used because of many local optima) and obtain a fitness score [Same as simple ES]
4. Keep only 10% of the highest fitness score coordinate pairs $(x, y)$
5. Randomly select 2 of these coordinate pairs $(x_1, y_1)$ and $(x_2, y_2)$ and randomly choose 1 $x$-coordinate and 1 $y$-coordinate from these pairs

Advantages:
- Maintain diversity (I'm guessing from Step 5?)

Disadvantages:
- Get stuck in local optima as well, especially because only the fittest are selected for the next generation
- $\sigma$ is still fixed so the range over which we're exploring values from does not vary
    - We want to search over a bigger space when we are uncertain that we're near a global optima and reduce search space when we are certain that we're near a global optima

More complex GAs:
- CoSyNe
- ESP
- NEAT



---
## Covariance-Matrix Adaptation Evolutionary Strategy (CMA-ES)<a id='cma-es'></a>

Summary:
- If our fittest individuals do not vary much, we're very close to the true global optima



1. Initialize $\mu^g = (\mu_x, \mu_y) = (0, 0)$ and **initialize** $\sigma = (\sigma_x, \sigma_y) = (1, 1)$
2. Sample $N$ coordinate pairs $(x, y)$ from multivariate gaussian with $\mu$ and $\sigma$ [Same as simple ES]
3. For each $(x, y)$ coordinate pair sampled, evaluate the function (e.g. Schaffer or Rastrigin - these were used because of many local optima) and obtain a fitness score [Same as simple ES]
4. Keep only 25% of the highest fitness score coordinate pairs $(x, y)$
5. Set new mean for next generation $\mu^{(g+1)} = (\frac{1}{N_{fittest 25\%}}\sum^{N_{fittest 25\%}}_{i=1} x_i, \frac{1}{N_{fittest 25\%}}\sum^{N_{fittest 25\%}}_{i=1} y_i)$ as the average of the fittest 25\% of the current population
6. Calculate new covariance matrix using current generation's mean:
$$
\begin{aligned}
\Sigma^{(g+1)} &= 
\begin{bmatrix}
    \frac{1}{N_{fittest 25\%}}\sum^{N_{fittest 25\%}}_{i=1} (x_i - \mu^g_x)^2 & \frac{1}{N_{fittest 25\%}}\sum^{N_{fittest 25\%}}_{i=1} (x_i - \mu^g_x)(y_i - \mu^g_y) \\
    \frac{1}{N_{fittest 25\%}}\sum^{N_{fittest 25\%}}_{i=1} (y_i - \mu^g_y)(x_i - \mu^g_x) & \frac{1}{N_{fittest 25\%}}\sum^{N_{fittest 25\%}}_{i=1} (y_i - \mu^g_y)^2 \\
\end{bmatrix}
\end{aligned}
$$
    - By calculating the next generation's covariance matrix $\Sigma^{(g+1)}$ using the current generation's mean $\mu^{g}$, we are intrinsically finding out whether the current generation's best 25\% are near the mean $\mu^g$. If they are near, the search space / Covariance matrix $\Sigma$ reduces in size because we are very near a global optima

Advantages:
- Increases search space when best solution is far away, and reduces search space when best solution is nearby.

Disadvantages:
- Covariance matrix computation: $O(N^2)$
- Use only when we have < 10K parameters



---
## Natural Evolution Strategy (REINFORCE-ES)<a id='nes'></a>

Summary:
- Don't eliminate any individuals, update them instead to maximize the average fitness score of a generation, not the total fitness score (previous methods eliminated the weak individuals to maximize overall fitness)



1. Initialize $\mu^g = (\mu_x, \mu_y) = (0, 0)$ and **initialize** $\sigma = (\sigma_x, \sigma_y) = (1, 1)$
2. Sample $N$ coordinate pairs $(x, y)$ from multivariate gaussian with $\mu$ and $\sigma$ [Same as simple ES]
3. For each $(x, y)$ coordinate pair sampled, evaluate the function (e.g. Schaffer or Rastrigin - these were used because of many local optima) and obtain a fitness score [Same as simple ES]
4. Compute the gradient update for $\mu$ and $\sigma$ by means of maximizing the log-likelihood of distribution we sample the coordinate pairs $(x, y)$ from
5. Update $\mu$ and $\sigma$ using their gradients

Advantages:
- $O(N) \because$ non-diagonal entries of $\Sigma$ do not need to be computed

Disadvantages:
- 



---
## OpenAI Evolution Strategy<a id='oais'></a>

Summary:
- Same as REINFORCE-ES, but do not update the covariance matrix
- Similar to simple ES



---
## Use of Fitness Shaping Functions

- To ensure a few, rare, outlier solutions with very high fitness scores don't dominate Natural Evolution strategies, after getting all the fitness scores of the generation, we rank them, and give them a relative fitness score instead of using their absolute score to calculate how much weight they have in the gradient updates.



---
### Resources:
- [David Ha's "A Visual Guide to Evolution Strategies"](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)

