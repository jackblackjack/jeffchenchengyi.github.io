---
interact_link: content/machine-learning/unsupervised-learning/bayesian-methods/markov-chains.ipynb
kernel_name: python3
has_widgets: false
title: 'Markov Chains'
prev_page:
  url: /machine-learning/unsupervised-learning/bayesian-methods/bayesian-learning
  title: 'Probabilistic Graphical Models'
next_page:
  url: /machine-learning/unsupervised-learning/bayesian-methods/hmm
  title: 'Hidden Markov Models'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Markov Chains

A Markov Chain is a discrete stochastic process with the *Markov property* : $P(X_t \mid X_{t−1}, \ldots,X_1)=P(X_t \mid X_{t−1})$. It is fully determined by a probability transition matrix $P$ which defines the transition probabilities $(P_{ij}=P(X_t=j \mid X_{t−1}=i)$ and an initial probability distribution specified by the vector $x$ where $x_i=P(X_0=i)$. The time-dependent random variable $X_t$ is describing the state of our probabilistic system at time-step $t$.

We'll go through markov chains using this example:
- Consider a world where a company A has 10% of the market share, meaning that other pest extermination companies have 90% of the market share. A is considering launching an ad campaign which they predict will have the following result:
    - People using other brands will **switch** to A with a probability of 0.6 within **one week** after seeing the ad
    - People already using A will **continue** using A with a probability of 0.8 within **one week** after seeing the ad
- Now we will use a Markov chain to model this process



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import numpy as np

```
</div>

</div>



---
# Initial State Distribution Matrix

$$
S_0 = 
\stackrel{\mbox{$A\,\,\,\,\,\neg A$} }
{\underset{1\,\times\,2}
{\begin{bmatrix}
  0.1 & 0.9 \\
\end{bmatrix} } }
$$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
S_0 = np.array([0.1, 0.9])
S_0

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([0.1, 0.9])
```


</div>
</div>
</div>



---
# Transition Probability Matrix

$$
P =
\text{Current State}\,
\begin{cases}
A \\
\neg A \\
\end{cases}
\overbrace{
\stackrel{\mbox{$A\,\,\,\,\,\neg A$} }
{
\underset{2\,\times\,2}{
\begin{bmatrix}
  0.8 & 0.2 \\
  0.6 & 0.4 \\
\end{bmatrix} } }
}^{\text{Next State} }
$$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
P = np.array([[0.8, 0.2], [0.6, 0.4]])
P

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0.8, 0.2],
       [0.6, 0.4]])
```


</div>
</div>
</div>



---
# After $1$ weeks...

After 1 week:
$$
\begin{aligned}
S_1 &= S_0 \cdot P^1 \\
&= \underset{1\,\times\,2}{\begin{bmatrix} 0.1 & 0.9 \\ \end{bmatrix} }
\cdot \underset{2\,\times\,2}{\begin{bmatrix} 0.8 & 0.2 \\ 0.6 & 0.4 \\ \end{bmatrix} } \\
&= \underset{1\,\times\,2}{\begin{bmatrix} 0.62 & 0.38 \\ \end{bmatrix} }
\end{aligned}
$$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
S_1 = np.dot(S_0, P)
S_1

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([0.62, 0.38])
```


</div>
</div>
</div>



---
# After $2$ weeks...

After 1 week:
$$
\begin{aligned}
S_1 &= S_0 \cdot P^2 \\
&= \underset{1\,\times\,2}{\begin{bmatrix} 0.1 & 0.9 \\ \end{bmatrix} }
\cdot \underset{2\,\times\,2}{\begin{bmatrix} 0.8 & 0.2 \\ 0.6 & 0.4 \\ \end{bmatrix} } \cdot \underset{2\,\times\,2}{\begin{bmatrix} 0.8 & 0.2 \\ 0.6 & 0.4 \\ \end{bmatrix} } \\
&= \underset{1\,\times\,2}{\begin{bmatrix} 0.724 & 0.276 \\ \end{bmatrix} }
\end{aligned}
$$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
S_2 = np.dot(np.dot(S_0, P), P)
S_2

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([0.724, 0.276])
```


</div>
</div>
</div>



---
# Stationary Matrix

If the probabilities in $P$ remain valid over a long period of time, what happens to the companies market share?

After $n$ weeks:
$$
\begin{aligned}
S_n &= S_0 \cdot P^n \\
&= \underset{1\,\times\,2}{\begin{bmatrix} 0.1 & 0.9 \\ \end{bmatrix} }
\cdot \underset{2\,\times\,2}{
{\begin{bmatrix} 0.8 & 0.2 \\ 0.6 & 0.4 \\ \end{bmatrix} }^n
} \\
&= \underset{1\,\times\,2}{\begin{bmatrix} 0.75 & 0.25 \\ \end{bmatrix} }
\end{aligned}
$$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
n = 100
S_n = np.dot(S_0, np.linalg.matrix_power(P, n))
S_n

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([0.75, 0.25])
```


</div>
</div>
</div>



The matrix $S_n$ is known as the Stationary matrix, and the system is said to be at steady state.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
np.linalg.matrix_power(P, n)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0.75, 0.25],
       [0.75, 0.25]])
```


</div>
</div>
</div>



Note that after a large number of steps the initial state does not matter any more, the probability of the chain being in any state $j$ is independent of where we started. This is our first view of the *equilibrium distribuion* of a Markov Chain. These are also known as the *limiting probabilities* of a Markov chain or *stationary distribution*.



---
# Questions

- Does every Markov Chain have a unique stationary matrix?
    - No, only for **regular** markov chains

- If a Markov chain has a unique stationary matrix, will the successive state matrices always approach this stationary matrix?
    - No, only for **regular** markov chains



---
# Regular Markov Chains

- A transition matrix $P$ is regular if some power of $P$ has only positive entries. A markov chain is a regular markov chain if its transition matrix is regular.
    - i.e. If you keep multiplying $P$ by itself and at some point you get all positive entries, then the matrix is regular
    
## Properties

Let $P$ be the transition matrix for a regular markov chain
1. There is a unique stationary matrix $\pi$ that can be found by solving the equation $\pi \cdot P = \pi$
- Given any initial state matrix $\pi_0$, the state matrices $\pi_k$ approach the stationary matrix $\pi$
- The matrices $P^k$ approach a limiting matrix $\bar{P}$, where each row of $\bar{P}$ is equal to the stationary matrix $\pi$



### 1. $\pi \cdot P = \pi$

E.g.
$$
\begin{aligned}
\pi \cdot P &= \pi \\
{\begin{bmatrix} s_1 & s_2 \end{bmatrix} } \cdot 
{\begin{bmatrix} 0.6 & 0.4 \\ 0.2 & 0.8 \end{bmatrix} }
&= {\begin{bmatrix} s_1 & s_2 \end{bmatrix} } \\
\vdots \\
0.6s_1 + 0.2s_2 &= s_1 \\
0.4s_1 + 0.8s_2 &= s_2 \\
s_1 + s_2 &= 1 \\
\vdots \\
s_1 &\approx 0.33 \\
s_2 &\approx 0.67 \\
\end{aligned}
$$



Notice again that the initial state matrix doesn't matter at all unless your markov chain isn't regular



---
# Absorbing Markov Chains

- A state in a markov chain is called an **absorbing state** if once the state is entered, it is impossible to leave.
    - i.e. entries on the diagonal, $p_{ii} = 1$, state i is an absorbing state

- A markov chain is an absorbing chain if:
    1. There is at least one absorbing state
    2. It is possible to go from each non-absorbing state to at least one absorbing state in a finite number of steps
        - There should be a set of arrows in a transition diagram from each node to an absorbing node
        
## Standard Form

- All absorbing states (will form an identity matrix) precede the non-absorbing states in the matrix

$$
P = 
\begin{aligned}
&A \\
&B \\
&C \\
&D \\
\end{aligned}
\stackrel{\mbox{$A\,\,\,\,\,\,\,\,\,B\,\,\,\,\,\,\,\,\,C\,\,\,\,\,\,\,\,\,D$} }
{
\begin{bmatrix}
    0.0 & 0.3 & 0.3 & 0.4 \\
    0.0 & 1.0 & 0.0 & 0.0 \\
    0.0 & 0.0 & 1.0 & 0.0 \\
    0.8 & 0.1 & 0.1 & 0.0 \\
\end{bmatrix}
}
\,
\underset{\text{Standardize} }{\rightarrow}
\,
\begin{aligned}
&B \\
&C \\
&A \\
&D \\
\end{aligned}
\stackrel{\mbox{$B\,\,\,\,\,\,\,\,\,C\,\,\,\,\,\,\,\,\,A\,\,\,\,\,\,\,\,\,D$} }
{
\begin{bmatrix}
    1.0 & 0.0 & 0.0 & 0.0 \\
    0.0 & 1.0 & 0.0 & 0.0 \\
    0.3 & 0.3 & 0.0 & 0.4 \\
    0.1 & 0.1 & 0.8 & 0.0 \\
\end{bmatrix}
}
$$

$$
I =
\begin{bmatrix}
    1.0 & 0.0 \\
    0.0 & 1.0
\end{bmatrix}
$$

$$
O =
\begin{bmatrix}
    0.0 & 0.0 \\
    0.0 & 0.0
\end{bmatrix}
$$

$$
R =
\begin{bmatrix}
    0.3 & 0.3 \\
    0.1 & 0.1
\end{bmatrix}
$$

$$
Q =
\begin{bmatrix}
    0.0 & 0.4 \\
    0.8 & 0.0
\end{bmatrix}
$$

If a standard form $P$ for an absorbing markov chain is partitioned as:
$$
P = 
\begin{bmatrix}
    I & O \\
    R & Q
\end{bmatrix} 
\,
\text{Standard Form},
$$
then $P^k$ approaches a limiting matrix $\bar{P}$ as $k$ increases, where
$$
\bar{P} =
\begin{bmatrix}
    I  & O \\
    FR & O
\end{bmatrix} 
\,
\text{and}
\,
F = {(I - Q)}^{-1},
\,
F\text{: Fundamental Matrix for }\,P
\,
$$

i.e. In the long run, $\bar{P}$:
$$
\bar{P} =
\begin{bmatrix}
    I  & O \\
    FR & O
\end{bmatrix} 
=
\begin{bmatrix}
    B \rightarrow B=1.0 & B \rightarrow C=0.0 & B \rightarrow A=0.0 & B \rightarrow D=0.0 \\
    C \rightarrow B=0.0 & C \rightarrow C=1.0 & C \rightarrow A=0.0 & C \rightarrow D=0.0 \\
    A \rightarrow B=0.5 & A \rightarrow C=0.5 & A \rightarrow A=0.0 & A \rightarrow D=0.0 \\
    D \rightarrow B=0.5 & D \rightarrow C=0.5 & D \rightarrow A=0.0 & D \rightarrow D=0.0 \\
\end{bmatrix}
$$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
P = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0], 
              [0.3, 0.3, 0.0, 0.4], 
              [0.1, 0.1, 0.8, 0.0]])

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
I = P[:2,:2]
I

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[1., 0.],
       [0., 1.]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
O = P[:2,2:]
O

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0., 0.],
       [0., 0.]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
R = P[2:,:2]
R

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0.3, 0.3],
       [0.1, 0.1]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
Q = P[2:,2:]
Q

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[0. , 0.4],
       [0.8, 0. ]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
F = np.linalg.inv(I - Q)
F

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[1.47058824, 0.58823529],
       [1.17647059, 1.47058824]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
P_bar = np.concatenate((np.concatenate((I, O), axis=1), 
                        np.concatenate((np.dot(F, R), O), axis=1)), axis=0)
P_bar

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[1. , 0. , 0. , 0. ],
       [0. , 1. , 0. , 0. ],
       [0.5, 0.5, 0. , 0. ],
       [0.5, 0.5, 0. , 0. ]])
```


</div>
</div>
</div>



---
# Ergodic Markov Chains

- Not every Markov Chain has a stationary distribution or even a unique one. But we can guarantee these properties if we add two additional constraints to the Markov Chain:
    1. ***Irreducible***: we must be able to reach any one state from any other state eventually (i.e. the expected number of steps is finite).
    2. ***Aperiodic***: the system never returns to the same state with a fixed period (e.g. not returning to start "sunny" deterministically every 5 steps).
    
- Together these two properties define the property ergodic. An important theorem says that if a Markov Chain is ergodic then it has a unique steady state probability vector $\pi$. In the context of MCMC, we can jump from any state to any other state (with some finite probability), trivially satisfying irreducibility.



---
# Detailed balance and Reversible Markov Chains

- A Markov Chain is said to be reversible (also known as the detailed balance condition) if there exists a probability distribution $\pi$ that satisfies this condition:

$$ \pi_i P(X_{n+1}=j \mid X_n=i)=\pi_j P(X_{n+1}=i\mid X_n=j) (3)$$

- In other words, in the long run, the proportion of times that you transition from state $i$ to state $j$ is the same as the proportion of times you transition from state $j$ to state $i$. In fact, if a Markov Chain is reversible then we know that it has a stationary distribution (which is why we use the same notation $\pi$).



---
## Resources:
- [patrickJMT on Markov Chains](https://www.youtube.com/watch?v=uvYTGEZQTEs)
- [Matthew Stephens on Markov Chains](https://stephens999.github.io/fiveMinuteStats/markov_chains_discrete_intro.html)
- [Brian Keng on MCMC](http://bjlkeng.github.io/posts/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm/)

