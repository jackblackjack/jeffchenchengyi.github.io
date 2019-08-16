---
interact_link: content/machine-learning/10-bayesian-methods/hmm.ipynb
kernel_name: python3
has_widgets: false
title: 'Hidden Markov Models'
prev_page:
  url: /machine-learning/10-bayesian-methods/markov-chains
  title: 'Markov Chains'
next_page:
  url: /machine-learning/06-natural-language-processing/basics/README
  title: 'Basics'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Hidden Markov Models



## Baum-Welch Algorithm

- Gives you both the most likely hidden transition probabilities as well as the most likely set of emission probabilities given only the observed states of the model (and, usually, an upper bound on the number of hidden states). You also get the "pointwise" highest likelihood points in the hidden states, which is often slightly different from the single hidden sequence that is overall most likely.



## Viterbi Algorithm

- If you know the transition probabilities for the hidden part of your model, and the emission probabilities for the visible outputs of your model, then the Viterbi algorithm gives you the most likely complete sequence of hidden states conditional on both your outputs and your model specification.

Baum-Welch VS Viterbi:
- If you know your model and just want the latent states, then there is no reason to use the Baum-Welch algorithm. If you don't know your model, then you can't be using the Viterbi algorithm.



---
# Connectionist Temporal Classification



---
# Kalman Filters and Weiner Filters



---
## Resources:
- [Difference between Baum-Welch and Viterbi](https://stats.stackexchange.com/questions/581/what-are-the-differences-between-the-baum-welch-algorithm-and-viterbi-training)
- [CMU HMM and Kalman Filter notes](https://www.cs.cmu.edu/~guestrin/Class/10701-S05/slides/hmms.pdf)
- [Examples of HMMs](https://www.math.unl.edu/~sdunbar1/ProbabilityTheory/Lessons/HiddenMarkovModels/Examples/examples.html)
- [Applications of HMMs](http://www.cs.umd.edu/~djacobs/CMSC828/ApplicationsHMMs.pdf)
- [Kalman Filters](https://www.kalmanfilter.net/default.aspx)

