---
interact_link: content/machine-learning/00-math-for-ml/linalg-de.ipynb
kernel_name: python3
has_widgets: false
title: 'Linear Algebra and Differential Equations'
prev_page:
  url: /machine-learning/00-math-for-ml/calculus
  title: 'Calculus'
next_page:
  url: /machine-learning/00-math-for-ml/statistics
  title: 'Statistics'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Linear Algebra and Differential Equations Review

We'll go through the linear algebra and differential equations topics you need to know to understand the theory behind machine learning algorithms.

### Table of Contents
1. [Linear Algebra](#linalg)
2. [Differential Equations](#de)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
# plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (18, 12)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (18, 12)

```
</div>

</div>



---
# Linear Algebra<a id='linalg'></a>



## Preliminaries

A [Hilbert Space](https://en.wikipedia.org/wiki/Hilbert_space#Definition) $H$ is a real / complex [inner product space](https://en.wikipedia.org/wiki/Inner_product_space) (A Vector Space with the inner product defined). [Eucliden Space](https://en.wikipedia.org/wiki/Euclidean_space) is a type of Hilbert Space and it is represented by the Real Coordinate Space $\mathcal{R}^n$, the set of all possible real-valued $n$-tuples (ordered sequence of real-valued scalars), where $n$ is the number of dimensions.



### Position / Location Vectors Vs. Spatial / Euclidean Vectors

In the real coordinate space $\mathcal{R}^n$, where $n$ is the number of dimensions, a **vector in standard position**, AKA a **position vector**, AKA a **location vector** $\mathbf{r} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$ is one that starts from the origin ${(0 , \ldots, 0)}$ and ends at the coordinates ${(x_1 , \ldots, x_n)}$. If however, the vector does not start at the standard position / origin, it is called a **spatial vector**, AKA a **euclidean vector**.



### Inner Products and Projections in $\mathcal{R}^n$

Euclidean / $l_2$ Norm: $\|\vec{x}\|_2 = \sqrt{x_1^2 + \cdots + x_n^2}$

Cosine similarity: $cos\theta = \frac{\vec{a} \cdot \vec{b} }{\|\vec{a}\|_2 \|\vec{b}\|_2}$

Standard Inner Product (Dot Product): $\vec{a} \cdot \vec{b} = \mathbf{a}^\top \mathbf{b}$

Inner Product: $\left\langle
    \begin{bmatrix} a_0 \\ \vdots \\ a_n \end{bmatrix},
    \begin{bmatrix} b_0 \\ \vdots \\ b_n \end{bmatrix}
    \right\rangle$
    
The Un-normalized Cosine Similarity = Dot Product, and the dot product is a special case of the Inner Product. There are [alternative inner products](http://mathonline.wikidot.com/vector-dot-product-euclidean-inner-product#toc4) that can be defined in Euclidean Space.

Projections:
- Scalar Projection of $\vec{b}$ onto $\vec{a}$, AKA component of $\vec{b}$ in direction of $\vec{a}$ $\rightarrow comp_{\vec{a} }\vec{b} = \|\vec{b}\|_2cos\theta = \frac{\vec{a} \cdot \vec{b} }{\|\vec{a}\|_2}$
- Vector Projection of $\vec{b}$ onto $\vec{a}$ is just the scalar projection multiplied by the unit vector of $\vec{a}$ $\rightarrow projection_{\vec{a} }\vec{b} = \|\vec{b}\|_2cos\theta * \frac{\vec{a} }{\|\vec{a}\|_2} = \frac{\vec{a} \cdot \vec{b} }{\|\vec{a}\|_2} * \frac{\vec{a} }{\|\vec{a}\|_2}$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.plot(np.arange(0, np.pi, 0.01), np.cos(np.arange(0, np.pi, 0.01)))
plt.grid()
plt.title('cos wave')
plt.show();

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/machine-learning/00-math-for-ml/linalg-de_7_0.png)

</div>
</div>
</div>



### Parametric Representation of Lines

We can represent a vector 



### Equation of a Hyperplane

Back in highschool, we learnt that a line in 2-D is denoted by 
$$
y=mx+b
$$
, where $y$ is the vertical axis, $x$ is the horizonatal axis, $m$ is the slope of the line, and $b$ is the y-intercept.

However, now in the world of linear algebra, we learn that the general form of any hyperplane (*A subspace of one dimension less than its ambient space, e.g. line in 2-D is a hyperplane, plane in 3-D is a hyperplane too*) is 
$$
{w_n}{x_n} + {w_{n - 1}{x_{n - 1} } } + ... + {w_0}{x_0} = 
\begin{bmatrix} {w_n}, {w_{n - 1} }, ..., {w_0} \end{bmatrix}
\begin{bmatrix} {x_n} \\ {x_{n - 1} } \\ \vdots \\ {x_0} \end{bmatrix} = \mathbf{w}^\top\mathbf{x} = 0
$$

Hence,
1. $$y=mx+b$$
2. $$y-mx-b=0$$
3. $$(1)*(y)+(-m)*(x)+(-b)*(1)=0$$
4. $$
\begin{bmatrix} 1, -m, -b \end{bmatrix}
\begin{bmatrix} y \\ x \\ 1 \end{bmatrix}=\mathbf{w}^\top\mathbf{x}=0
$$

From now onwards, whenever we're talking about hyperplanes, we can always translate all the data points horizontally and vertically such that the hyperplane passes through the origin. This is a very important point as it allows us to fix this form $\rightarrow \mathbf{w}^\top\mathbf{x} = 0$ as the equation of a hyperplane. Furthermore, it much better sense now when we talk about why all the points on the hyperplane, $\forall{i} \vec{x_i}$, are orthogonal to $\vec{w}$, making their dot product / inner product / un-normalized cosine similarity $= 0$.



## Singular Value Decomposition (SVD)



## QR Decomposition (QR)



## LU Decomposition



---
# Differential Equations<a id='de'></a>

Differential equations are broken down into 2 types:
1. Ordinary Differential Equations (ODE)
    - Derivatives w.r.t. only 1 independent variable
2. Partial Differential Equations (PDE)
    - Derivatives w.r.t. multiple independent variables



## 1st Order Equations



## 2nd Order Equations



## Graphical and Numerical Methods



## Vector Spaces and Subspaces



## Eigenvalues and Eigenvectors



## Applied Mathematics and $A^\top A$



## Fourier and Laplace Transforms



---
## Resources:
- [Why $\mathbf{X}^T \mathbf{X}$ is always positive semi-definite](https://statisticaloddsandends.wordpress.com/2018/01/31/xtx-is-always-positive-semidefinite/)
- [Dot Products and Projections](https://math.oregonstate.edu/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/dotprod/dotprod.html)
- [Understanding what $\mathbf{w}$ is and why $y =mx + b$ <=> $\mathbf{w}^\top\mathbf{x}=0$](https://www.youtube.com/watch?v=3qzWeokRYTA)
- [Learn Differential Equations: Up Close with Gilbert Strang and Cleve Moler on MIT Opencourseware](https://ocw.mit.edu/resources/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/index.htm)

