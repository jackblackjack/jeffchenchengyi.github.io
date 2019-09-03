---
redirect_from:
  - "/machine-learning/02-unsupervised-learning/clustering/readme"
title: 'Clustering'
prev_page:
  url: /machine-learning/01-supervised-learning/estimation/convex-optimization
  title: 'Convex Optimization'
next_page:
  url: /machine-learning/02-unsupervised-learning/clustering/k-means
  title: 'K-Means'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---
# Clustering

<img src='https://2.bp.blogspot.com/-gKsHFQmvz_0/Vwe64nJdSII/AAAAAAAA2eY/nzWsfZESRG0ZSNRGlfa6ASqDdRJgKzt0A/s1600/output_3BAiEC.gif' style='border: 5px solid black; border-radius: 5px;'/>

Ways to categorize Clustering Method Types:

1. Goal
    1. Monothetic: Cluster members have the same common property
        - e.g. Cluster 1 contains only males taller than 178cm, but can differ in many other features
    2. Polythetic: Cluster members are **similar** to each other (Similarity defined using a distance such as [Minkowski distances](https://en.wikipedia.org/wiki/Minkowski_distance))
        - e.g. K-means - samples with closest euclidean distances to one another cluster together
2. Overlap
    1. Hard Clustering: Clusters do not overlap
        - Elements either belong in the cluster or not
    2. Soft Clustering: Clusters may overlap
        - Each element has a "strength of association" score with its cluster such a probability that the element is found in the cluster
3. Flat or Hierarchical
    1. Set of Groups
        - Clusters of different animals like penguin, shark, dog
    2. Taxonomy
        - Clusters of bird, fish, mammal
        
### Table of Contents
1. [$k$-means](https://jeffchenchengyi.github.io/machine-learning/02-unsupervised-learning/clustering/k-means.html)
2. [Gaussian Mixture Models](https://jeffchenchengyi.github.io/machine-learning/02-unsupervised-learning/clustering/gmm.html)