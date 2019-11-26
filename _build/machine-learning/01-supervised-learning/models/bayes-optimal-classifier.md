---
interact_link: content/machine-learning/01-supervised-learning/models/bayes-optimal-classifier.ipynb
kernel_name: python3
has_widgets: false
title: 'Bayes Optimal Classifier'
prev_page:
  url: /machine-learning/01-supervised-learning/models/README.html
  title: 'Models'
next_page:
  url: /machine-learning/01-supervised-learning/models/nn.html
  title: 'K-Nearest Neighbours'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Bayes Optimal Classifier



The Bayes Optimal Classifier assumes you know the posterior distribution $\mathrm{P}(y\mid\mathbf{x})$. For example, say you've seen the entire universe of SPAM emails and have derived the conditional probabilities as such for a particular sentence (set of features):

$$
\mathrm{P}(\text{SPAM}\mid \mathbf{x} = \text{"There are currently 30 hot single ladies in your area waiting to see you!"})=0.8\\
\mathrm{P}(\text{NOT SPAM}\mid \mathbf{x} = \text{"There are currently 30 hot single ladies in your area waiting to see you!"})=0.2\\
$$

In other words, 80% of all emails in the universe that only contained "There are currently 30 hot single ladies in your area waiting to see you!" are in fact spam, while 20% of the rest were not spam, but actually legitimate information, lol.

In this case, the bayes optimal classifier will predict the most likely label given the features, hence predicting SPAM for all the times when it sees an email to contain solely "There are currently 30 hot single ladies in your area waiting to see you!".

This means that the classifier will get 20% of the predictions wrong and have an error rate of 0.2.



## Best constant predictor

Let's also consider what's the worst possible classifier we can build, the constant predictor. This classifier always predicts the same value - the most frequent label - independent of the features of each observation. In other words, the classifier is just returning the mode for classification and in regression, it'll just return the mean if the loss function is squared loss and median if loss function is absolute loss.



---
## Resources:
- [Kilian Weinberger's kNN lectures](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)

