---
interact_link: content/machine-learning/08-genetic-algorithms/hbpso-ehw.ipynb
kernel_name: python3
has_widgets: false
title: 'Human Behavior Particle Swarm Optimization on Hardware Configuration'
prev_page:
  url: /machine-learning/08-genetic-algorithms/novelty-search.html
  title: 'Novelty Search'
next_page:
  url: /machine-learning/00-math-for-ml/README.html
  title: 'Math for Machine Learning'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Evolvable Hardware with Human Behavior Particle Swarm Optimization (HBPSO)



---
## Particle Swarm Optimization (PSO)

In particle swarm optimization, each of our individuals (just like in simple genetic algorithms) are known as particles and the entire optimization process is modelled to mimic bee swarms / flocks of birds. If we're trying to **minimize** a function with $D$ dimensions, in each iteration $k$ of PSO, each particle $i \in n$ has:

1. Particle position: $x^i_{k} \in \mathbb{R}^D$
2. Particle velocity: $v^i_{k} \in \mathbb{R}^D$
3. Best individual particle's position (Best parameters of an individual particle over its past iterations): $p^i_k \in \mathbb{R}^D$
4. Best swarm position: $p^g_k \in \mathbb{R}^D$
5. Constant inertia weight: $w_k \in \mathbb{R}$
6. Cognitive and Social Parameters: $c_1, c_2 \in \mathbb{R}$
7. Random weights: $r_1, r_2 \in [0, 1]$

After each iteration, updates are as follows for each particle:

1. Velocity: $v^i_{k+1} = w_kv^i_{k} + \underbrace{c_1r_1(p^i_k - x^i_k)}_{\text{Social Term}} + \underbrace{c_2r_2(p^g_k - x^i_k)}_{\text{Cognitive Term}}$
2. Position: $x^i_{k+1} = x^i_{k} + v^i_{k+1}$

More [here](https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/)



---
## Human Behavior PSO

Very similar to PSO, except that we not only incorporate information about the global best position found by the swarm so far, but also the worst position found so far (mimic how humans learn bad behaviors when surrounded by people with bad behavior) and social and cognitive parameters are reduced to just random coefficient scalars. Updates on velocity of particle will become:

1. Velocity: $v^i_{k+1} = w_kv^i_{k} + \underbrace{r_1(p^i_k - x^i_k)}_{\text{Social Term}} + \underbrace{r_2(p^{\text{swarm's best}}_k - x^i_k)}_{\text{Cognitive Term (Good influence)}} + \underbrace{r_3(p^\text{swarm's worst}_k - x^i_k)}_{\text{Cognitive Term (Bad influence)}}$



---
## Evolving Circuit Design

- We encode circuit designs as circuit matrices:

$$
\text{Inputs} 
\rightarrow 
\underbrace{
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
}_{\text{Circuit Matrix}}
\rightarrow
\text{Outputs}
$$

- Each $a_{ij}$ represents the < Where input is from >< {AND, OR, XOR, NOT, WIRE} >< Where output will go >

Algorithm:
1. Initialize a population / swarm of random circuit matrices representing the configuration of the proposed circuit
2. Evaluation of evolved circuits and compare with desired circuits
3. Everytime an output from circuit is equal to output from truth table, fitness increases by 1

Experiment Results:
- Number of columns in Circuit matrix heavily impacts algorithms performance in reducing time complexity, but rows are not very important



---
### Resources:
- [Nathan Rooy on Particle Swarm Optimization from Scratch with Python](https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/)
- [Evolvable Hardware Design of Digital Circuits using the New Human Behavior Based Particle Swarm Optimization](https://www.researchgate.net/publication/281837677_Evolvable_Hardware_Design_of_Digital_Circuits_using_the_New_Human_Behavior_Based_Particle_Swarm_Optimization)

