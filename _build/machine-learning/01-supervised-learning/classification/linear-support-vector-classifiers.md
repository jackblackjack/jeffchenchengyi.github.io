---
interact_link: content/machine-learning/01-supervised-learning/classification/linear-support-vector-classifiers.ipynb
kernel_name: python3
has_widgets: false
title: 'Linear Support Vector Classifiers (SVC)'
prev_page:
  url: /machine-learning/01-supervised-learning/classification/perceptron
  title: 'Perceptron'
next_page:
  url: /machine-learning/01-supervised-learning/classification/naivebayes
  title: 'Naive Bayes Classifier'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Support Vector Classifiers

In this algorithm cookbook, we'll go over the support vector classifiers algorithm.

*Before we go on, remember that in coordinate geometry, we can always translate all the data points horizontally and vertically in all dimensions such that our linearly separating hyperplane passes through THE ORIGIN. Therefore, as a pre-processing step, we will always translate all our data points so that our hyperplane passes throught the origin, and all the points on the hyperplane, $\mathbf{x} = \begin{bmatrix} {x_n} \\ {x_{n - 1} } \\ \vdots \\ {x_0} \end{bmatrix}$, will be orthogonal to the $\mathbf{w}$ which passes through the origin.*

Similar to the Perceptron algorithm, our hyperplane is defined as 

$$h(x_i) = \textrm{sign}(\mathbf{w}^\top \mathbf{x}_i + b) \rightarrow \textrm{sign}(\mathbf{w}^\top \mathbf{x}_i) \text{ by making } \mathbf{x}_i \leftarrow \begin{bmatrix} \mathbf{x}_i \\ 1  \end{bmatrix} \text{ and } \mathbf{w} \leftarrow \begin{bmatrix} \mathbf{w} \\ b  \end{bmatrix} \\ $$ (*$h$ stands for hyopothetis and this "absorbing" of the bias term just increases the dimension that we're working on by 1, e.g. originally our data was in 2-D, and now it's shifted in the z-axis to live in 3-D*)

![adding_dimension_perceptron][adding_dimension_perceptron]

[adding_dimension_perceptron]: http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/Perceptron/PconstantDim.png "adding_dimension_perceptron"

However, in contrast to the Perceptron that finds A hyperplane that linearly separates the data if it exists, SVM finds the __MAXIMUM MARGIN__ separating hyperplane.

![svm][svm]

[svm]: http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/svm/margin.png "svm"

We define the hyperplane to be:
$$\mathcal{H}=\left\{\mathbf{x}\vert{}\mathbf{w}^T\mathbf{x}+b=0\right\}=\left\{\mathbf{x}\vert{}\mathbf{w}^T\mathbf{x}=0\right\}$$ (*After the addition of one dimension*)
This means that all points $\mathbf{x}$ on the hyperplane that are orthogonal to $\mathbf{w}^T$ so that their dot product is 0.



---
# Brief Overview

Intuition:
- Similar to the Perceptron, it finds a linearly separating hyperplane for the dataset. However, it finds one not through iteration and adjusting the $\mathbf{w}$ (hyperplane). It jumps straight to the unique hyperplane that maximizes the margin between the hyperplane and the support vectors (closest datapoints to hyperplane).

What is it used for?
- Classification

Assumption: 
- It is a binary classification problem
- Data is linearly separable



---
# Training

Training the SVM essentially involves solving a quadratic programming problem (*a type of constrained optimization problem (commonly solved using lagrangian multipliers*) to find the __simplest__ hyperplane that maximizes the margin.

## SVM with Hard Constraints
In order to setup the constrained optimization problem, we have to first define the margin:
1. What is the distance of a point $\mathbf{x}$ to the hyperplane?
    - Considering the image below (however, translating the hyperplane denoted by the solid black line to the origin by absorbing the bias term $b$), we see that the vector $\mathbf{d}$ is simply the component of $\mathbf{x}$ in the direction of the unit vector of $\mathbf{w}$.
    - $\therefore \text{Scalar Projection of } \mathbf{x} \text{ onto } \mathbf{w} = \|\mathbf{d}\|_2 = \frac{\mathbf{w}\cdot\mathbf{x} }{\|\mathbf{w}\|_2} = \frac{\mathbf{w}^\top\mathbf{x} }{\|\mathbf{w}\|_2} = {\|\mathbf{x}\|_2}cos\theta$
    - We define margin of the hyperplane as the minimum of this
        - $\gamma(\mathbf{w})=\min_{\mathbf{x}\in D}\frac{\left | \mathbf{w}^\top \mathbf{x}\right |}{\left \| \mathbf{w} \right \|_{2} } = |{\|\mathbf{x}\|_2}cos\theta|$
2. Constrained Optimization Objective:
$$\underbrace{\max_{\mathbf{w} }\gamma(\mathbf{w})}_{maximize \ margin}  \textrm{such that} \ \  \underbrace{\forall i \ y_{i}(\mathbf{w}^\top x_{i})\geq 0}_{separating \ hyperplane}$$ (constraint means that all data points are correctly classified).
    - Substituting $\gamma(\mathbf{w})=\min_{\mathbf{x}\in D}\frac{\left | \mathbf{w}^\top \mathbf{x}\right |}{\left \| \mathbf{w} \right \|_{2} }$, we get 
$$\underbrace{\max_{\mathbf{w} }\underbrace{\frac{1}{\left \| \mathbf{w} \right \|}_{2}\min_{\mathbf{x}_{i}\in D}\left | \mathbf{w}^\top\mathbf{x}_{i} \right |}_{\gamma(\mathbf{w})} \ }_{maximize \ margin} \ \  s.t. \ \  \underbrace{\forall i \ y_{i}(\mathbf{w}^\top x_{i})\geq 0}_{separating \ hyperplane}$$
    - We can now rescale $\mathbf{w}$ so that $\min_{\mathbf{x}\in D}\left | \mathbf{w}^\top \mathbf{x} \right |=\|\mathbf{w}\|\|\mathbf{x}\||\underbrace{cos\theta}_{-1 \leq cos\theta \leq 1}|=1$
    - New objective:
        1. $\max_{\mathbf{w} }\frac{1}{\left \| \mathbf{w} \right \|_{2} }\cdot 1 = \min_{\mathbf{w} }\left \| \mathbf{w} \right \|_{2} = \min_{\mathbf{w} } \mathbf{w}^\top \mathbf{w} 
        \begin{align}
        &\textrm{s.t. } 
        \begin{matrix}
        \forall i, \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i})&\geq 0\\ 
        \min_{i}\left | \mathbf{w}^\top \mathbf{x}_{i} \right | &= 1
        \end{matrix}
        \end{align}$
        2. $\min_{\mathbf{w} }\mathbf{w}^\top \mathbf{w} \ \textrm{s.t.} \forall i \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) \geq 1$
            - $\text{(A)}\rightarrow\text{(B)}$:
                - If all points are correctly classified by hyperplane, AKA, $\forall i, \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) \geq 0$, and minimum $|\mathbf{w}^\top \mathbf{x}| = |\|\mathbf{w}\|\|\mathbf{x}\|\underbrace{cos\theta}_{-1 \leq cos\theta \leq 1}| = 1$, $\forall i \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) \geq 1$.
            - $\text{(B)}\rightarrow\text{(A)}$:
                - If $\forall i \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) \geq 1$, I must definitely classify all my points correctly, and since I'm greater than equal to 1, and $y_i$ can only take either $\{+1, -1\}$, minimum $\mathbf{w}^\top \mathbf{x}_{i}$ is definitely 1.

$$\therefore \min_{\mathbf{w} }\mathbf{w}^\top \mathbf{w} = \min_{\mathbf{w} }\|\mathbf{w}\|^2_2 = {w_n}^2 + ... + {w_1}^2 + {w_0}^2 \ \textrm{s.t.} \forall i \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) \geq 1$$ is a quadratic optimization problem to __find the simplest hyperplane (where simpler means smaller $\mathbf{w}^\top \mathbf{w}$) such that all inputs lie at least 1 unit away from the hyperplane on the correct side__.

    
![distance_svm][distance_svm]

[distance_svm]: http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/svm/projection.png "distance_svm"

## SVM with Soft Constraints
Sometimes it is not possible to linearly separate our data not because there is indeed an underlying non-linear structure, but because there exists outliers and anomalies in each class. Hence, it'll be nice if we can find a hyperplane that __*bests*__ linearly separates our data, forgiving a few points that might be outliers and are causing SVM to find a hyperplane with a really tight margin. To do this, we introudce a slack variable $\xi_i$ for each data point, transforming the constrained optimization problem:

$$
\begin{matrix}
\min_{\mathbf{w} }\mathbf{w}^\top \mathbf{w}
\ : \forall i \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) \geq 1
\end{matrix}
\rightarrow
\begin{matrix}
\min_{\mathbf{w} }\mathbf{w}^\top \mathbf{w}+C\sum_{i=1}^{n}\xi _{i}\ 
: \forall i \ y_{i}(\mathbf{w}^\top \mathbf{x}_{i})\geq 1-\xi_i
\end{matrix}
$$

- $C$ is a hyperparameter that needs to be tuned - the cost per magnitude of slack variable added

$$
\xi_i=\left\{
\begin{array}{cc}
\ 1-y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) & \textrm{ if $y_{i}(\mathbf{w}^\top \mathbf{x}_{i})<1$}\\
0 & \textrm{ if $y_{i}(\mathbf{w}^\top \mathbf{x}_{i})\geq 1$}
\end{array}
\right
\}
=\max(1-y_{i}(\mathbf{w}^\top \mathbf{x}_{i}) ,0)
$$

![soft_svm][soft_svm]

[soft_svm]: http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/svm/figure.png "soft_svm"

- Recall that the margin is defined as $\gamma(\mathbf{w})=\min_{\mathbf{x}\in D}\frac{\left | \mathbf{w}^\top \mathbf{x}\right |}{\left \| \mathbf{w} \right \|_{2} } = |{\|\mathbf{x}\|_2}cos\theta|$. Hence, SVM tries to find the support vector (a data point $x_i$ used to dictate the margin) that makes the largest angle that's < 90 degrees to the $\mathbf{w}$ vector as this minimizes the $|{\|\mathbf{x}\|_2}cos\theta|$.
- As you can see, as your $C$ hyperparameter decreases, the margin gets looser because it becomes much more cheap to add the slack variable, making SVM choose the support vectors (e.g. when $C=0.1$, the data point chosen as the support vector is at (1, 2), making $\theta \approx 30 \text{ degrees}$) that did not make the largest angle < 90 degrees with the $\mathbf{w}$ vector in cases with higher $C$ values (e.g. when $C=100$, the data point chosen as the support vector is at (-1, 0), making $\theta \approx 60 \text{ degrees}$).

- Rules of thumb:
    - A small $C$ will give a wider margin, at the cost of some misclassifications.
    - A huge $C$ will give the hard margin classifier and tolerates zero constraint violation.
    - The key is to find the value of such that noisy data does not impact the solution too much.
    
Furthermore, with the addition of the slack variable, our __constrained__ optimization problem can be setup as an __unconstrained__ optimization problem:
$$
\min_{\mathbf{w} }\underbrace{\mathbf{w}^\top \mathbf{w} }_{l_{2}-regularizer}+        C\  \sum_{i=1}^{n}\underbrace{\max\left [ 1-y_{i}(\mathbf{w}^\top \mathbf{x}),0 \right ]}_{hinge-loss} \label{eq:svmunconst}
$$

- This formulation allows us to optimize the SVM paramters $\mathbf{w}$ just like logistic regression (e.g. through gradient descent). The only difference is that we have the hinge-loss instead of the logistic loss.



---
# Prediction

Similar to the Perceptron, we can just plug in ${\mathbf{x}_{test} }$ with the trained / fitted ${\mathbf{w}^\top}$ in ${\hat{y} = \mathbf{w}^\top \cdot \mathbf{x}_{test} }$



---
## Resources:
- [Kilian Weinberger's Linear SVM lecture](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote09.html)
- [SVM and Lagrange Multiplier](https://towardsdatascience.com/understanding-support-vector-machine-part-1-lagrange-multipliers-5c24a52ffc5e)
- [Alexandre KOWALCZYK's e-book on SVMs](https://www.svm-tutorial.com/svm-tutorial/)
- [Understanding what $\mathbf{w}$ is and why $y =mx + b$ <=> $\mathbf{w}^\top\mathbf{x}=0$](https://www.youtube.com/watch?v=3qzWeokRYTA)

