---
interact_link: content/machine-learning/08-genetic-algorithms/es-vs-rl.ipynb
kernel_name: python3
has_widgets: false
title: 'Evolutionary Strategies Vs. Reinforcement Learning'
prev_page:
  url: /machine-learning/08-genetic-algorithms/evolutionary-strategies.html
  title: 'Evolutionary Strategies'
next_page:
  url: /machine-learning/08-genetic-algorithms/novelty-search.html
  title: 'Novelty Search'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Evolution Strategies Vs. Reinforcement Learning



---
## Reinforcement Learning
$Action \sim Distribution(\theta) \rightarrow Parameters$

1. Define a policy function
    - A mapping of actions agent can take in a particular state of its environment
2. Train Policy
    1. Randomly intialize probabilities of each action given state
    2. Record all sequence of actions (episodes of interaction) taken in reaction to environment state 
    3. Use backpropagation to update neural network weights to favor actions (increase probabilities of good actions) that had a reward
    
    
- Actions are directly drawn from a probability distribution, hence noisy
- Parameters (Neural Net weights) are deterministic given actions $\because$ gradients are deterministically computed



---
## Evolution Strategies



$Parameters \sim Distribution(\theta) \rightarrow Action$

1. Randomly initialize parameter vectors
2. Evaluate policy network using those parameter vectors
3. Set new parameter vector to weighted sum of original vectors, weighing by how much reward it generated (more reward, greater the weight)


- Parameters are drawn from probability distribution, no gradients computed, just random subset of parameters are chosen from search space to see which best maximizes a given objective function
- Actions are dependent on the parameters

Advantages:
- No need backpropagation
- Parallelizable
- Robustness
    - ES global optima does not differ based on episodes of interaction rate)
- Structured Exploration
    - Actions are deterministic, meaning that the policies are deterministic
- Credit assignment over long time scales
    - ES works well even with long episodes of interaction
    
Disadvantages:
- If noisy parameters don't lead to different outcomes, we won't get a gradient signal, and we can't train the network



---
### Resources:
- [OpenAI on Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://openai.com/blog/evolution-strategies/)

