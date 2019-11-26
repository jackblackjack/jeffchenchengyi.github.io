---
interact_link: content/machine-learning/01-supervised-learning/models/nn.ipynb
kernel_name: python3
has_widgets: false
title: 'K-Nearest Neighbours'
prev_page:
  url: /machine-learning/01-supervised-learning/models/bayes-optimal-classifier.html
  title: 'Bayes Optimal Classifier'
next_page:
  url: /machine-learning/01-supervised-learning/models/perceptron.html
  title: 'Perceptron'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Nearest Neighbor Algorithms



---
# Brief Overview

What is it used for?
- Classification
- Regression

Assumption: 
- Data points that are 'near' to each other have the same class.

Distance Measures: 
- Minkowski
$${\text{dist}(\mathbf{x},\mathbf{z})=\left(\sum_{r=1}^d |x_r-z_r|^p\right)^{1/p} }$$
    1. ${p \to 1:}$ Manhatten / Taxicab / L1 distance
    2. ${p \to 2:}$ Euclidean / L2 distance
    3. ${p \to \infty:}$ Maximum distance
    
What does it do?
- 1-nearest neighbour partitions space into voronoi diagram where each line segment represents 



---
# Training

We don't train the k-NN explicitly. We memorize all the data and only during the predicting phase do we do something.



---
# Prediction

1. Choose a distance / similarity metric
2. Go through the ${n}$ training data points and calculate distance / similarity from / to data point to be predicted
3. Sort by smallest distance first / highest similarity first $O(n)$ OR maintain a min-heap so that we can extract the shortest distance point in $O(log(n))$ and insert in $O(log(n))$ OR fixed-size binary heap if we already know $k$
4. Take the ${k}$ smallest distances / highest similarity data points and take the most common label as the class of current data point.



---
# Problems

Curse of Dimensionality:
- As the number of features / dimensions grows, the amount of data we need to generalize accurately grows exponentially.
- However, the data might only live on a sub-manifold of the dimension, like a contorted hyperplane in 3-D space.



---
# Implementation

- 1-D: Use the vanilla approach as stated above.
- 2-D to 8-D: Use K-D Trees
- 9-D and above: Use Approximate Nearest Neighbours



---
# Disadvantages

- Doesn't work well in high dimensional spaces



---
## Resources:
- [Kilian Weinberger's k-NN lecture](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)
- [Kevin Zakka's k-NN blog](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/)
- [David Thompson on Curse of Dimensionality](https://www.youtube.com/watch?v=dZrGXYty3qc)
- [Victor Lavrenko on Nearest Neighbour Methods](https://www.youtube.com/watch?v=k_7gMp5wh5A&list=PLBv09BD7ez_68OwSB97WXyIOvvI5nqi-3&index=1)
- [Jonathan Richard Shewchuk's notes on Nearest Neighbour Algorithms](https://people.eecs.berkeley.edu/~jrs/189/lec/25.pdf)

