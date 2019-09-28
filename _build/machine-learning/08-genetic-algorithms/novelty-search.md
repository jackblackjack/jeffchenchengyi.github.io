---
interact_link: content/machine-learning/08-genetic-algorithms/novelty-search.ipynb
kernel_name: python3
has_widgets: false
title: 'Novelty Search'
prev_page:
  url: /machine-learning/08-genetic-algorithms/es-vs-rl
  title: 'Evolutionary Strategies Vs. Reinforcement Learning'
next_page:
  url: /machine-learning/08-genetic-algorithms/hbpso-ehw
  title: 'Human Behavior Particle Swarm Optimization on Hardware Configuration'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Evolutionary Strategies + Directed Exploration Algos

- Here we will go through improving Evolutionary strategies by combining it with algorithms that promote directed exploration like Quality Diversity and Novelty Search to improve performance by avoiding local minima during learning.



---
## Novelty Search for RL

What do we have at each generation:
1. Policy $\pi \sim Distribution(\theta)$: e.g. $\pi \sim \mathcal{N}(\mu, \sigma)$, $\therefore$ Each policy has a different probability of being activated in the population
2. Because the policy comes from a probability distribution, the behavior of the individuals $b(\pi)$ also follow a distribution
3. An archive set $A$
4. A probability distribution that determines whether $b(\pi)$ will be added to the archive set $A$
5. Novelty score $N$ of a policy is essentially how far away the behaviors that result from that policy diverge from each other
    - We compute that particular policy $\pi_\theta$'s Novelty by running $k$-NN on $b(\pi_\theta)$ and the rest of the behaviors that were in the archive set $A$
    - Find the average distance between the $k$ nearest neighbors of behavior $b(\pi_\theta)$ and that becomes the novelty score



---
## Hybrid Strategies

### Novelty Search + Evolutionary Strategy (NS-ES)
- Update Policy $\pi$'s probability distribution according and kill off individuals depending on their relative novelty score of the individual's behavior
    1. We will use a "relative novelty score" (Individual's novelty divided by the sum of all the novelties of all individuals / policy) as the probability of being selected into the next generation
    2. The greater the raw novelty score of the individual, the more we will update the policy's probability distribution's parameters

### Quality Diversity + Evolutionary Strategy

#### Novelty Search + Reward (Fitness) + Evolutionary Strategy (NSR-ES)
- Use **simple average** of the fitness score for each individual and raw novelty score to update the parameters
- The fittest and novel individuals help update the parameters the most
- Implication: An individual who's fit but not really novel has equal say as an individual who's not fit but really novel in how much to update the parameters to favor their probabilities to be replicated in the next generation

#### Novelty Search + Reward (Fitness) + Adapt + Evolutionary Strategy (NSRA-ES)
- Use a **weighted average** of the fitness score for each individual and raw novelty score to update the parameters instead
- Start off with a fixed $w$ for the fitness score and $(1-w)$ for raw novelty score's weights and **increase** when the overall performance of the population increases
    - This means that once our population has a very high fitness, we don't really care about diversity and hence, novelty that much anymore
    - Novelty is a way to increase our search space for the best individuals similar to CMA-ES, when we're updating the covariance matrix.



---
### Resources:
- [Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents](https://arxiv.org/pdf/1712.06560.pdf)

