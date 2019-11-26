---
interact_link: content/machine-learning/00-math-for-ml/linalg-de.ipynb
kernel_name: python3
has_widgets: false
title: 'Linear Algebra and Differential Equations'
prev_page:
  url: /machine-learning/00-math-for-ml/calculus.html
  title: 'Calculus'
next_page:
  url: /machine-learning/00-math-for-ml/probability-statistics.html
  title: 'Probability Theory and Statistics'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Linear Algebra and Differential Equations Review

We'll go through the linear algebra and differential equations topics you need to know to understand the theory behind machine learning algorithms.

### Table of Contents
1. [Linear Algebra](#linalg)
    1. [Preliminaries](#prelim)
    2. [Types of Systems of Linear Equations](#sletypes)
    3. [Ways to solve Systems of Linear Equations](#waystosolvesle)
    4. [Matrix Factorization](#matfact)
2. [Differential Equations](#de)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (18, 12)
get_colors = lambda length: plt.get_cmap('Spectral')(np.linspace(0, 1.0, length))

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



## Preliminaries<a id='prelim'></a>

A [Hilbert Space](https://en.wikipedia.org/wiki/Hilbert_space#Definition) $H$ is a real / complex [inner product space](https://en.wikipedia.org/wiki/Inner_product_space) (A Vector Space with the inner product defined). [Eucliden Space](https://en.wikipedia.org/wiki/Euclidean_space) is a type of Hilbert Space and it is represented by the Real Coordinate Space $\mathcal{R}^n$, the set of all possible real-valued $n$-tuples (ordered sequence of real-valued scalars), where $n$ is the number of dimensions.



### Position / Location Vectors Vs. Spatial / Euclidean Vectors

In the real coordinate space $\mathcal{R}^n$, where $n$ is the number of dimensions, a **vector in standard position**, AKA a **position vector**, AKA a **location vector** $\mathbf{r} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$ is one that starts from the origin ${(0 , \ldots, 0)}$ and ends at the coordinates ${(x_1 , \ldots, x_n)}$. If however, the vector does not start at the standard position / origin, it is called a **spatial vector**, AKA a **euclidean vector**.



### Inner Products and Projections in $\mathcal{R}^n$

Euclidean / $l_2$ Norm: $\vert\vert\vec{x}\vert\vert_2 = \sqrt{x_1^2 + \cdots + x_n^2}$

Cosine similarity: $cos\theta = \frac{\vec{a} \cdot \vec{b} }{\vert\vert\vec{a}\vert\vert_2 \vert\vert\vec{b}\vert\vert_2}$

Standard Inner Product (Dot Product): $\vec{a} \cdot \vec{b} = \mathbf{a}^\top \mathbf{b}$

Inner Product: $\left\langle
    \begin{bmatrix} a_0 \\ \vdots \\ a_n \end{bmatrix},
    \begin{bmatrix} b_0 \\ \vdots \\ b_n \end{bmatrix}
    \right\rangle$
    
The Un-normalized Cosine Similarity = Dot Product, and the dot product is a special case of the Inner Product. There are [alternative inner products](http://mathonline.wikidot.com/vector-dot-product-euclidean-inner-product#toc4) that can be defined in Euclidean Space.

Projections:
- Scalar Projection of $\vec{b}$ onto $\vec{a}$, AKA component of $\vec{b}$ in direction of $\vec{a}$ $\rightarrow comp_{\vec{a} }\vec{b} = \vert\vert\vec{b}\vert\vert_2cos\theta = \frac{\vec{a} \cdot \vec{b} }{\vert\vert\vec{a}\vert\vert_2}$
- Vector Projection of $\vec{b}$ onto $\vec{a}$ is just the scalar projection multiplied by the unit vector of $\vec{a}$ $\rightarrow projection_{\vec{a} }\vec{b} = \vert\vert\vec{b}\vert\vert_2cos\theta * \frac{\vec{a} }{\vert\vert\vec{a}\vert\vert_2} = \frac{\vec{a} \cdot \vec{b} }{\vert\vert\vec{a}\vert\vert_2} * \frac{\vec{a} }{\vert\vert\vec{a}\vert\vert_2}$



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



### Standard Form of Equation for Hyperplane

Back in highschool, we learnt that a line in 2-D Cartesian plane is denoted by 
$$
y=mx+b
$$
, where $y$ is the vertical axis, $x$ is the horizonatal axis, $m$ is the slope of the line, and $b$ is the y-intercept.

However, now in the world of linear algebra, we learn that the **standard form** of any hyperplane (*A subspace of one dimension less than its ambient space, e.g. line in 2-D is a hyperplane, plane in 3-D is a hyperplane too*) is 

$$
{w_n}{x_n} + {w_{n - 1}{x_{n - 1} } } + ... + {w_0}{x_0} = 
\begin{bmatrix} {w_n}, {w_{n - 1} }, ..., {w_0} \end{bmatrix}
\begin{bmatrix} {x_n} \\ {x_{n - 1} } \\ \vdots \\ {x_0} \end{bmatrix} = \mathbf{w}^\top\mathbf{x} = b\,\text{, where }b=0\,\text{if hyperplane passes through the origin}
$$

In general, given an $N$-dimensional space, we need $N-P$ equations like $\sum_i^n w_i x_i = b$ to define an object of $P$ dimensions.

Hence, in 2-D Cartesian plane, to define a 1-D line, we need just $(N=2) - (P=1) = 1$ equation:
$$
\begin{aligned}
y&=mx+b\,\text{, assuming}\,b \not= 0 \\
y-mx&=b \\
(-m)*(x)+(1)*(y)&=b \\
\begin{bmatrix} -m, 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} &= b \\
\mathbf{w}^\top\mathbf{x} &= b
\end{aligned}
$$

Notice that from the same equation, if we push the non-zero b into the LHS, we can create a hyperplane that passes through the origin but doing this increases our dimensions by 1:
$$
\begin{aligned}
y&=mx+b \\
y-mx-b&=0 \\
(1)*(y)+(-m)*(x)+(-b)*(1)&=0 \\
\begin{bmatrix} 1, -m, -b \end{bmatrix}
\begin{bmatrix} y \\ x \\ 1 \end{bmatrix}&=\mathbf{w}^\top\mathbf{x}=0
\end{aligned}
$$

Because of this, we can always translate all the data points horizontally and vertically such that the hyperplane passes through the origin, however increasing our dimensionality by 1. 

*This is a very important point as it allows us to fix this form $\rightarrow \mathbf{w}^\top\mathbf{x} = 0$ as the equation of a hyperplane. Furthermore, it much better sense now when we talk about why all the points on the hyperplane, $\forall{i} \vec{x_i}$, are orthogonal to $\vec{w}$, making their dot product / inner product / un-normalized cosine similarity $= 0$. $\mathbf{w}$ is also known as the **normal** vector and it uniquely defines the hyperplane.*
- To find the normal vector $\mathbf{w}$, we need to cross two **spatial** vectors that are in the hyperplane (unless our hyperplane passes through the origin, which means we can either use **spatial** or **position** vectors since both lie on the plane):
    - Given that 3 points lie on a plane: $p_0 = \begin{bmatrix} 1, 2, 3 \end{bmatrix}$, $p_1 = \begin{bmatrix} 4, -1, 2 \end{bmatrix}$, $p_2 = \begin{bmatrix} 2, 0, 4 \end{bmatrix}$, we can't just do the cross product of 2 of the points to find the normal vector as those position vectors are not on the hyperplane because the hyperplane does not pass through the origin. We therefore need to find the spatial vectors on the hyperplane by subtracting any 2 unique combinations of the points on hyperplane / position vectors $p_0, p_1, p_2$ and crossing them to get the normal. This example is shown below:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Vector coordinates
p0, p1, p2, origin = \
    np.array([1,2,3]), \
    np.array([4,-1,2]), \
    np.array([2,0,4]), \
    np.array([0,0,0])

# Normal Vector for plane passing through p0, p1, p2
# by getting cross product of spatial vectors p1-p0
# and p2-p0 that lie on the plane
normal = np.cross(p1-p0, p2-p0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = get_colors(7)

# Plot vectors
p0_quiver = ax.quiver(*origin, *p0, color=colors[0], label='$p_0$')
p1_quiver = ax.quiver(*origin, *p1, color=colors[1], label='$p_1$')
p2_quiver = ax.quiver(*origin, *p2, color=colors[2], label='$p_2$')
p0p1_cross_quiver = ax.quiver(*origin, *np.cross(p0, p1), color=colors[3], label='$p_0\,x\,p_1$')
p1_p0_quiver = ax.quiver(*p0, *p1-p0, color=colors[4], label='$p_1-p_0$')
p2_p0_quiver = ax.quiver(*p0, *p2-p0, color=colors[5], label='$p_2-p_0$')
normal_quiver = ax.quiver(*origin, *normal, color=colors[6], label='Normal')

# Annotate vectors
ax.text(*origin, s='{}'.format('Origin: '+str(origin)))
ax.text(*p0, s='{}'.format('$p_0$: '+str(p0)))
ax.text(*p1, s='{}'.format('$p_1$: '+str(p1)))
ax.text(*p2, s='{}'.format('$p_2$: '+str(p2)))
ax.text(*np.cross(p0, p1), s='{}'.format('$p_0\,x\,p_1$: '+str(list(np.cross(p0, p1)))))
ax.text(*p1+1, s='{}'.format('$p_1 - p_0$: '+str(p1-p0)))
ax.text(*p2+1, s='{}'.format('$p_2 - p_0$: '+str(p2-p0)))
ax.text(*normal, s='{}'.format('Normal: '+str(normal)))

# Plot plane
# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -p0.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(10), range(10))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
ax.plot_surface(xx, yy, z, alpha=0.2)

# Set viewing perspective
ax.view_init(10, -30)

# Set axis limits
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# Set labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../../images/machine-learning/00-math-for-ml/linalg-de_9_0.png)

</div>
</div>
</div>



In order to get the cartesian / standard equation for that hyperplane, we find the dot product of the normal vector with any of the points on the hyperplane / position vectors to get the $b$ of the standard form of a hyperplane equation.

Our Normal Vector: $\mathbf{w}^\top = \begin{bmatrix}-5, -4, -3\end{bmatrix}$

Our $b$: $\mathbf{w}^\top p_0 = \begin{bmatrix}-5 & -4 & -3\end{bmatrix} \begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix} = (-5 * 1) + (-4 * 2) + (-3 * 3) = - 5 - 8 - 9 = - 22$

Our final **standard form** of hyperplane equation:
$$
P = \{\mathbf{x} \vert \mathbf{w}^\top \mathbf{x} = \begin{bmatrix}5 & 4 & 3\end{bmatrix} \begin{bmatrix}x_0 \\ x_1 \\ x_2\end{bmatrix} = 5x_0 + 4x_1 + 3x_2 = 22\}
$$

If we absorb the $b$, $P$ is a 2D object living in 4D:
$$
P = \{\mathbf{x} \vert \mathbf{w}^\top \mathbf{x} = \begin{bmatrix}5 & 4 & 3 & -22 \end{bmatrix} \begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ 1 \end{bmatrix} = 5x_0 + 4x_1 + 3x_2 - 22 = 0\}
$$

Meaning of $\mathbf{w}^\top$: For different values of $b$, scalar multiples of $\mathbf{w}^\top$ will get us hyperplanes that are parallel to our original hyperplane.

Meaning of the $b$: Recall the cosine similarity definition of dot products - $\mathbf{w}^\top \mathbf{x} = \vert\vert\mathbf{w}^\top\vert\vert_2 \vert\vert\mathbf{x}\vert\vert_2 cos\theta = b$
- This means that the **standard form** of the equation of hyperplane is saying that all the position vectors / points on the hyperplane have the same "un-normalized cosine similarity".
- If we normalize $b$ by dividing it by $\vert\vert\mathbf{w}^\top\vert\vert_2$, we get $\vert\vert\mathbf{x}\vert\vert_2 cos\theta = \frac{b}{\vert\vert\mathbf{w}^\top\vert\vert_2 \vert\vert}$, or the component of position vector $\mathbf{x}$ in the direction of the unit normal / the plane's distance from the origin along its unit normal.



### Parametric Form of Equation for Hyperplane

The **parametric form** of equation for hyperplane is a general position vector that is a linear combination of 2 things:
1. Position Vector $\mathbf{p}_0$ / Point in hyperplane
2. Basis of the same hyperplane but passing through origin ($P$ direction / spatial vectors $\mathbf{d}_i$ that lie on the hyperplane for a hyperplane of $P$ dimensionality)

$$
\begin{aligned}
\mathbf{x} &= \mathbf{p}_0 + \alpha_0\mathbf{d}_0 + \alpha_1\mathbf{d}_1 + \ldots + \alpha_{P-1}\mathbf{d}_{P-1} \\
&= \mathbf{p}_0 + \sum_{i=0}^{P-1}\alpha_i\mathbf{d}_i \\
&= \mathbf{p}_0 + \begin{bmatrix} \vert\vert & \vert\vert & \vert\vert & \vert\vert \\ \mathbf{d}_0 & \mathbf{d}_1 & \ldots & \mathbf{d}_{P-1} \\ \vert\vert & \vert\vert & \vert\vert & \vert\vert \\ \end{bmatrix} \vec{\alpha} \\
&= \mathbf{p}_0 + D\vec{\alpha},\,\vec{\alpha}\in\mathbb{R}^P
\end{aligned}
$$

$\vec{\alpha}$ is a vector of free variables.

Following from the example in the section above, if we have 3 points lie on a plane: $p_0 = \begin{bmatrix} 1, 2, 3 \end{bmatrix}$, $p_1 = \begin{bmatrix} 4, -1, 2 \end{bmatrix}$, $p_2 = \begin{bmatrix} 2, 0, 4 \end{bmatrix}$, what's the parametric form of the hyperplane?

Since we already have 3 points on the plane / position vectors to the plane, our job is already 50% completed. We need to find a basis for the plane. Recall that in order to find the normal vector, we found 2 spatial vectors that lie on the hyperplane by $p_1 - p_0 = \begin{bmatrix} 4 \\ -1 \\  2 \end{bmatrix} - \begin{bmatrix} 1 \\ 2 \\  3 \end{bmatrix} = \begin{bmatrix} 3 \\ -3 \\ -1 \end{bmatrix}$ and $p_2 - p_0 = \begin{bmatrix} 2 \\ 0 \\  4 \end{bmatrix} - \begin{bmatrix} 1 \\ 2 \\  3 \end{bmatrix} = \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix}$. Hence, our parametric form is:

$$
\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 3 & 1 \\ -3 & -2 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} \alpha_0 \\ \alpha_1 \end{bmatrix} = \left\{\begin{array}{lr}
x &= 1 + 3\alpha_0 + \alpha_1 \\
y &= 2 - 3\alpha_0 - 2\alpha_1 \\
z &= 3 - 1\alpha_0 + \alpha_1 \\
\end{array}\right\},\,\alpha_i\in\mathbb{R}
$$

OR 

$$
\mathbf{x} = \begin{bmatrix} 4 \\ -1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 & 1 \\ -3 & -2 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} \alpha_0 \\ \alpha_1 \end{bmatrix} = \left\{\begin{array}{lr}
x &= 4 + 3\alpha_0 + \alpha_1 \\
y &= - 1 - 3\alpha_0 - 2\alpha_1 \\
z &= 2 - 1\alpha_0 + \alpha_1 \\
\end{array}\right\},\,\alpha_i\in\mathbb{R}
$$

OR 

$$
\mathbf{x} = \begin{bmatrix} 2 \\ 0 \\ 4 \end{bmatrix} + \begin{bmatrix} 3 & 1 \\ -3 & -2 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} \alpha_0 \\ \alpha_1 \end{bmatrix} = \left\{\begin{array}{lr}
x &= 2 + 3\alpha_0 + \alpha_1 \\
y &= 0 - 3\alpha_0 - 2\alpha_1 \\
z &= 4 - 1\alpha_0 + \alpha_1 \\
\end{array}\right\},\,\alpha_i\in\mathbb{R}
$$



### [Invertible Matrix Theorem](https://www.math.dartmouth.edu/archive/m22f06/public_html/imt.pdf)

Let $A$ be an $n × n$ matrix. Then the following are equivalent:
1. The matrix $A$ is invertible (non-singular).
- The matrix $A$ is row equivalent to $I_n$.
- The matrix $A$ has $n$ pivot positions.
- The equation $A\mathbf{x} = 0$ has only the trivial solution.
- The columns of $A$ form a linearly independent set.
- The linear transformation $\mathbf{x} \rightarrow A\mathbf{x}$ is one-to-one.
- For each $\mathbf{b} \in \mathbb{R}^n$, the equation $A\mathbf{x} = b$ has a unique solution.
- The columns of $A$ span $\mathbb{R}^n$.
- The linear transformation $\mathbf{x} \rightarrow A\mathbf{x}$ is onto.
- There is an $n × n$ matrix $C$ such that $CA = I_n$.
- There is an $n × n$ matrix $D$ such that $AD = I_n$.
- The matrix $A^\top$ is invertible.
- The columns of $A$ form a basis for $\mathbb{R}^n$.
- The column space of $A$ is $\mathbb{R}^n$ (Col $A$ = $\mathbb{R}^n$).
- The dimension of the column space of $A$ is $n$ (dim Col $A$ = $n$).
- The rank of $A$ is $n$ (rank $A$ = $n$).
- The null space of $A$ is $\{0\}$ ($N(A)$ = $\{0\}$).
- The dimension of the null space of $A$ is 0 (dim $N(A)$ = 0).
- The number 0 is not an eigenvalue of $A$.
- The determinant of $A$ is not zero (det $A \not=$ 0).
- The orthogonal complement of the column space of $A$ is $\{0\}$ (${(Col A)}^\perp = \{0\}$).
- The orthogonal complement of the null space of $A$ is $\mathbb{R}^n$ (($N(A)^\perp = \mathbb{R}^n$).
- The row space of $A$ is $\mathbb{R}^n$ (Row $A$ = $\mathbb{R}^n$).
- The matrix $A$ had $n$ non-zero singular values.



### Rank-Nullity Theorem

$$
Rank(A) + Nullity(A) = n
$$



## Special Matrices

When a matrix is applied onto a vector, 2 things can happen.
1. Rotation
2. Stretching



### Unitary Matrix
Special Case: Rotation Matrix
- Rotates a vector

$$
\begin{bmatrix}
cos\theta & -sin\theta \\
sin\theta & cos\theta \\
\end{bmatrix}
$$



### Diagonal Matrix
- Stretches a vector

$$
\begin{bmatrix}
\alpha & 0 \\
0 & \beta \\
\end{bmatrix}
$$



### Coefficient Matrix

Given a system of linear equations $\mathbf{Ax = b}$, the coefficient matrix is:

$$
\mathbf{A} = 
\left(\begin{array}{ccccc}
a_{00} & a_{01} & a_{02} & \ldots & a_{0n} \\
a_{10} & a_{11} & a_{12} & \ldots & a_{1n} \\
\vdots & \vdots & \vdots & \ldots & \vdots \\
a_{m0} & a_{m1} & a_{m2} & \ldots & a_{mn} \\
\end{array}\right)
$$



### Augmented Matrix

Given a system of linear equations $\mathbf{Ax = b}$, the augmented matrix is:

$$
(\mathbf{A}\vert\mathbf{b}) = 
\left(\begin{array}{ccccc|c}
a_{00} & a_{01} & a_{02} & \ldots & a_{0n} & b_0 \\
a_{10} & a_{11} & a_{12} & \ldots & a_{1n} & b_1 \\
\vdots & \vdots & \vdots & \ldots & \vdots & \vdots \\
a_{m0} & a_{m1} & a_{m2} & \ldots & a_{mn} & b_m \\
\end{array}\right)
$$



### Orthogonal Matrix



### Orthonormal Matrix

- All column vectors in orthonormal matrix are orthogonal (perpendicular) to each other and are unit vectors (length = 1)



## Properties of Systems of Linear Equations $\underset{m x n}{\mathbf{A}}\,\underset{nx1}{\mathbf{x}} = \underset{mx1}{\mathbf{b}}$<a id='sleproperties'></a>



### Linear Independence

#### Row Space of $(\mathbf{A}\vert\mathbf{b})$ Perspective:
- No equations are scalar multiples of each other.
- Geometrically, this would mean that the system has not specified scalar multiples of the exact same hyperplane.

#### Column Space of $\mathbf{A}$ Perspective:
- No column vectors are scalar multiples of each other.
- Geometrically, the vectors of the column space span $\mathbb{R}^n$ because all column vectors are linearly independent, forming a basis for $\mathbb{R}^n$
- If all column vectors in $\mathbf{A}$ are linearly independent but column vectors in $(\mathbf{A}\vert\mathbf{b})$ are not, it means that there exists a solution for $\mathbf{Ax = b}$



### Consistency

#### Row Space of $(\mathbf{A}\vert\mathbf{b})$ Perspective:
- No equations contradict each other, meaning that there isn't a linear combination of 
- Geometrically, this would mean that the system has not specified scalar multiples of the exact same hyperplane.



### [Equivalence](https://en.wikipedia.org/wiki/System_of_linear_equations)

--- Under Construction ---



## Types of Systems of Linear Equations<a id='sletypes'></a>

We can categorize Systems of Linear Equations in 2 different ways:
1. Overdetermined Vs. Completely Determined Vs. Underdetermined
2. Homogeneous Vs. Inhomogeneous



### [Overdetermined System](https://en.wikipedia.org/wiki/Overdetermined_system)





### [Underdetermined System](https://en.wikipedia.org/wiki/Underdetermined_system)

--- Under Construction ---



### [Rouche-Capelli Theorem](https://en.wikipedia.org/wiki/Rouch%C3%A9%E2%80%93Capelli_theorem)

--- Under Construction ---



### Homogeneous System

$$
\begin{aligned}
\underset{m x n}{\mathbf{A}}\,\underset{nx1}{\mathbf{x}} &= \underset{mx1}{\mathbf{0}} \\
\begin{bmatrix}
    a_{00} & a_{01} & a_{02} & \ldots & a_{0n} \\
    a_{10} & a_{11} & a_{12} & \ldots & a_{1n} \\
    \vdots & \vdots & \vdots & \ldots & \vdots \\
    a_{m0} & a_{m1} & a_{m2} & \ldots & a_{mn} \\
\end{bmatrix}
\begin{bmatrix}
    x_0 \\
    x_1 \\
    \vdots \\
    x_n
\end{bmatrix} &=
\begin{bmatrix}
    0_0 \\
    0_1 \\
    \vdots \\
    0_m
\end{bmatrix}
\end{aligned}
$$

#### Goal (Column Space Perspective):
Find a linear combination of vectors in the column space $C(A)$ such that they sum to the zero vector $\vec{0}$:

$$
\begin{aligned}
\begin{bmatrix}
    \mathbf{c}_0 =
    \begin{bmatrix}
        a_{00} \\ a_{10} \\ \vdots \\ a_{m0}
    \end{bmatrix} &
    \mathbf{c}_1 =
    \begin{bmatrix}
        a_{01} \\ a_{11} \\ \vdots \\ a_{m1}
    \end{bmatrix} &
    \mathbf{c}_2 =
    \begin{bmatrix}
        a_{02} \\ a_{12} \\ \vdots \\ a_{m2}
    \end{bmatrix} &
    \ldots &
    \mathbf{c}_n =
    \begin{bmatrix}
        a_{0n} \\ a_{1n} \\ \vdots \\ a_{mn}
    \end{bmatrix} &
\end{bmatrix}
\begin{bmatrix}
    x_0 \\
    x_1 \\
    \vdots \\
    x_n
\end{bmatrix} &=
\begin{bmatrix}
    0_0 \\
    0_1 \\
    \vdots \\
    0_m
\end{bmatrix} \\
x_0\mathbf{c}_0 + x_1\mathbf{c}_1 + x_2\mathbf{c}_2 + \ldots + x_n\mathbf{c}_n &= 
\begin{bmatrix}
    0_0 \\
    0_1 \\
    \vdots \\
    0_m
\end{bmatrix} \\
\end{aligned}
$$

#### Goal (Row Space Perspective):
Find the cartesian coordinates of the intersection point(s) between all equations defined by the row space:

$$
\begin{aligned}
\begin{bmatrix}
    \mathbf{r}_0 =
    \begin{bmatrix}
        a_{00} & a_{01} & \ldots & a_{0n}
    \end{bmatrix} \\
    \mathbf{r}_1 =
    \begin{bmatrix}
        a_{10} & a_{11} & \ldots & a_{1n}
    \end{bmatrix} \\
    \mathbf{r}_2 =
    \begin{bmatrix}
        a_{20} & a_{21} & \ldots & a_{2n}
    \end{bmatrix} \\
    \vdots \\
    \mathbf{r}_n =
    \begin{bmatrix}
        a_{m0} & a_{m1} & \ldots & a_{mn}
    \end{bmatrix} \\
\end{bmatrix}
\begin{bmatrix}
    x_0 \\
    x_1 \\
    \vdots \\
    x_n
\end{bmatrix} &=
\begin{bmatrix}
    0_0 \\
    0_1 \\
    \vdots \\
    0_m
\end{bmatrix} \\
\begin{bmatrix}
\mathbf{r}_0 \cdot \mathbf{x} \\
\mathbf{r}_1 \cdot \mathbf{x} \\
\mathbf{r}_2 \cdot \mathbf{x} \\
\vdots \\
\mathbf{r}_m \cdot \mathbf{x} \\
\end{bmatrix} &=
\begin{bmatrix}
    0_0 \\
    0_1 \\
    \vdots \\
    0_m
\end{bmatrix} \\
\end{aligned}
$$

#### Trivial Solution:
- In the Column Space perspective, choosing none of the column vectors $\mathbf{c}_i$ (allocating a coefficient of $x_j = 0$ to each $\mathbf{c}_i$) gets us the $\vec{0}$, making the $\vec{0}$ a trivial solution to the homogeneous system of linear equations.
- In the Row Space perspective, each equation $\mathbf{r}_i \cdot \mathbf{x} = 0$ in the row space $R(A)$ the graphs for each equation passes through the origin. Hence all of them intersect at the origin, making $\vec{0}$ a trivial solution to the homogeneous system of linear equations.

#### Non-trivial Solution:




### Inhomogeneous System

$$
\mathbf{Ax = b},\,\mathbf{b}\not=0
$$



## Ways to solve Systems of Linear Equations<a id='waystosolvesle'></a>

Currently, we have 3 ways to solve systems of linear equations:
1. Gaussian and Gauss-Jordan Elimination
2. Cramer's Rule
3. LU Decomposition
4. LDU Decomposition



### 1. [Gaussian and Gauss-Jordan Elimination and Reduced Row Echelon Form (RREF)](https://www.freetext.org/Introduction_to_Linear_Algebra/Systems_Linear_Equations/Gaussian_and_Gauss-Jordan_Elimination/)

Gaussian and Gauss-Jordan Elimination Elementary Row operations:
1. Interchanging two rows ($R_k \leftrightarrow R_j$).
2. Adding a multiple of one row to another ($R_k \rightarrow R_k + \alpha R_j$).
3. Multiplying any row by a nonzero scalar value ($R_k \rightarrow \alpha R_k$).

Row Echelon Form (REF):
1. All zero rows are at the bottom of the matrix.
- If a pivot is defined as the first non-zero entry of any given row, then the pivot in each row after the first occurs at least 1 column further to the right than the previous row.
- The pivot in any nonzero row is 1.
- All entries in the column above and below a pivot are zero. (Necessary condition for **Reduced** Row Echelon Form (RREF))

We use Gaussian and Gauss-Jordan Elimination Elementary Row operations in order to get the REF and RREF forms of the matrix which will reveal important details. A few of the use cases are below:

#### Case 1: Matrix's invertibility
--- Under Construction ---

#### Case 2: Finding solutions to homogeneous and inhomogeneous systems of linear equations
--- Under Construction ---

#### Case 3: Finding an inverse of a matrix
--- Under Construction ---
$$
\left(\begin{array}{ccc|ccc}
  1 & 2 & 9 & 1 & 0 & 0\\
  3 & \pi & 6 & 0 & 1 & 0 \\
1/4 & 0 & 0 & 0 & 0 & 1
\end{array}\right)
$$



### 2. Cramer's Rule

Given an inhomogeneous system $\mathbf{Ax = b}$, a unique solution $\mathbf{x}$ exists **iff** $\mathbf{A}$ is invertible. Cramer's rule can be used to find this unique solution as an alternative to using Gaussian and Gauss-Jordan Elimination.

Rule: 
$$
x_j = \frac{det(\mathbf{A}_j)}{det(\mathbf{A})}
$$



### 3. [LU Decomposition](https://www.freetext.org/Introduction_to_Linear_Algebra/Systems_Linear_Equations/LU_Decomposition/)



### 4. LDU Decomposition



## Matrix Factorization<a id='matfact'></a>

Design / Data Matrix:
$$
\begin{aligned}
\underset{n\times m}{\mathbf{X}} &= \underset{n\,\text{samples}\,\times \,m\,\text{features}}{\begin{bmatrix} x_{11} & x_{12} & \ldots & x_{1m} \\ x_{21} & x_{22} & \ldots & x_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \ldots & x_{nm} \\ \end{bmatrix}}
\end{aligned}
$$

Unit Matrix (Matrix of all ones):
$$
\begin{aligned}
\underset{n\times n}{\mathbf{e}\mathbf{e}^\top} &= \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix} \cdot \begin{bmatrix} 1 & 1 & \ldots & 1 \end{bmatrix} \\
&= \underset{n\,\times \,n}{\begin{bmatrix} 1 & 1 & \ldots & 1 \\ 1 & 1 & \ldots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \ldots & 1 \\ \end{bmatrix}}
\end{aligned}
$$

Matrix of Feature / Covariate Means:
$$
\begin{aligned}
\underset{n\times m}{\bar{\mathbf{X}}} &= \frac{1}{n}\cdot\mathbf{e}\mathbf{e}^\top\cdot \underset{n\times m}{\mathbf{X}} \\
&= \underset{n\,\text{duplicates of feature means}\,\times \,m\,\text{features}}{\begin{bmatrix} \bar{x}_{1} & \bar{x}_{2} & \ldots & \bar{x}_{m} \\ \bar{x}_{1} & \bar{x}_{2} & \ldots & \bar{x}_{m} \\ \vdots & \vdots & \ddots & \vdots \\ \bar{x}_{1} & \bar{x}_{2} & \ldots & \bar{x}_{m} \\ \end{bmatrix}}
\end{aligned}
$$

Sample Data ($\frac{1}{n - 1}$) Covariance Matrix:
$$
\begin{aligned}
\underset{m\times m}{\Sigma} &= \frac{1}{n - 1} {(\underset{n\times m}{\mathbf{X}} - \underset{n\times m}{\bar{\mathbf{X}}})}^\top \cdot {(\underset{n\times m}{\mathbf{X}} - \underset{n\times m}{\bar{\mathbf{X}}})} \\
&= \frac{1}{n - 1} \begin{bmatrix} x_{11} - \bar{x}_{1} & x_{21} - \bar{x}_{1} & \ldots & x_{n1} - \bar{x}_{1} \\ x_{12} - \bar{x}_{2} & x_{22} - \bar{x}_{2} & \ldots & x_{n2} - \bar{x}_{2} \\ \vdots & \vdots & \ddots & \vdots \\ x_{1m} - \bar{x}_{m} & x_{2m} - \bar{x}_{m} & \ldots & x_{nm} - \bar{x}_{m} \\ \end{bmatrix} \cdot \begin{bmatrix} x_{11} - \bar{x}_{1} & x_{12} - \bar{x}_{2} & \ldots & x_{1m} - \bar{x}_{m} \\ x_{21} - \bar{x}_{1} & x_{22} - \bar{x}_{2} & \ldots & x_{2m} - \bar{x}_{m} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} - \bar{x}_{1} & x_{n2} - \bar{x}_{2} & \ldots & x_{nm} - \bar{x}_{m} \\ \end{bmatrix} \\
&= \frac{1}{n - 1} \begin{bmatrix} 
\sum^{n}_{i = 1} {(x_{i1} - \bar{x}_{1})}{(x_{i1} - \bar{x}_{1})} & \sum^{n}_{i = 1} {(x_{i1} - \bar{x}_{1})}{(x_{i2} - \bar{x}_{2})} & \ldots & \sum^{n}_{i = 1} {(x_{i1} - \bar{x}_{1})}{(x_{im} - \bar{x}_{m})} \\ 
\sum^{n}_{i = 1} {(x_{i2} - \bar{x}_{2})}{(x_{i1} - \bar{x}_{1})} & \sum^{n}_{i = 1} {(x_{i2} - \bar{x}_{2})}{(x_{i2} - \bar{x}_{2})} & \ldots & \sum^{n}_{i = 1} {(x_{i2} - \bar{x}_{2})}{(x_{im} - \bar{x}_{m})} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\sum^{n}_{i = 1} {(x_{im} - \bar{x}_{m})}{(x_{i1} - \bar{x}_{1})} & \sum^{n}_{i = 1} {(x_{im} - \bar{x}_{m})}{(x_{i2} - \bar{x}_{2})} & \ldots & \sum^{n}_{i = 1} {(x_{im} - \bar{x}_{m})}{(x_{im} - \bar{x}_{m})} \\ \end{bmatrix} \\
&= \begin{bmatrix} 
Var(x_1) & Cov(x_1, x_2) & \ldots & Cov(x_1, x_m) \\ 
Cov(x_2, x_1) & Var(x_2) & \ldots & Cov(x_2, x_m) \\
\vdots & \vdots & \ddots & \vdots \\
Cov(x_m, x_1) & Cov(x_m, x_2) & \ldots & Var(x_m) \\ 
\end{bmatrix}
\end{aligned}
$$



### [Eigendecomposition / Spectral Decomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Eigendecomposition_of_a_matrix)

Let $\Sigma$ be a square $m \times m$ matrix with $m$ linearly independent eigenvectors $q_{i = 1, \ldots, m}$ (E.g. Covariance Matrix $\because \Sigma \in S_{+}^m$), we can find the eigendecomposition of $\Sigma$ by finding:

$$
\begin{aligned}
\underset{m \times m}{\Sigma} &= \underset{m \times m}{Q}\cdot\underset{m \times m}{\Lambda}\cdot\underset{m \times m}{Q^{-1}} \\
&= \underbrace{\begin{bmatrix} \vert & \vert & \ldots & \vert \\ q_1 & q_2 & \ldots & q_m \\ \vert & \vert & \ldots & \vert \end{bmatrix}}_{\text{Orthogonal Matrix}} \cdot \begin{bmatrix} \lambda_i & 0 & \ldots & 0 \\ 0 & \lambda_2 & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & \lambda_m \end{bmatrix} \cdot \underbrace{\begin{bmatrix} \vert & \vert & \ldots & \vert \\ q_1 & q_2 & \ldots & q_m \\ \vert & \vert & \ldots & \vert \end{bmatrix}^{-1}}_{\text{Orthogonal Matrix}}
\end{aligned}
$$



### [Singular Value Decomposition (SVD)](https://www.youtube.com/watch?v=mBcLRGuAFUk&t=47s)

Every $$

$$
\begin{aligned}
\underset{n \times m}{A} &= \underset{n \times m}{U}\cdot\underset{m \times m}{\Sigma}\cdot\underset{m \times n}{V^\top} \\
&= 
\underbrace{\begin{bmatrix} \vert & \vert & \ldots & \vert \\ u_1 & u_2 & \ldots & u_m \\ \vert & \vert & \ldots & \vert \end{bmatrix}}_\text{Rotation} 
\cdot 
\underbrace{\begin{bmatrix} \sigma_i & 0 & \ldots & 0 \\ 0 & \sigma_2 & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & \sigma_m \end{bmatrix}}_\text{Stretching} 
\cdot 
\underbrace{\begin{bmatrix} \vert & \vert & \ldots & \vert \\ v_1 & v_2 & \ldots & v_m \\ \vert & \vert & \ldots & \vert \end{bmatrix}^\top}_\text{Rotation}
\end{aligned}
$$



$$
\begin{aligned}

\end{aligned}
$$



### [QR Decomposition (QR)](https://en.wikipedia.org/wiki/QR_decomposition)



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



## Applied Mathematics and [Gramian Matrix $G = X^\top X$](https://en.wikipedia.org/wiki/Gramian_matrix)

#### $X^\top X$ is always Positive Semi-definite (PSD) $G \in \mathbf{S}_+$

#### $X^\top X$ is always Symmetric:

$$
{(X^\top X)}^\top = X^\top {(X^\top)}^\top = X^\top X \because {(X^\top)}^\top = X\,\text{and}\,{(XB)}^\top = B^\top X^\top
$$

#### [Inverse of a Symmetric Matrix $X$ is another Symmetric Matrix $X^{-1}$](https://www.quora.com/What-is-the-inverse-of-a-symmetric-matrix):

$$
\begin{aligned}
X^{-1}X &= I \\
{X^{-1}X}^\top &= I^\top \\
X^\top {(X^{-1})}^\top &= I \\
X {(X^{-1})}^\top &= I \because X = X^\top \\
\therefore {(X^{-1})}^\top &= X^{-1},\,\text{making inverse of symmetric matrix also symmetric}\,\because\,\text{inverse of matrix is unique}
\end{aligned}
$$



## Fourier and Laplace Transforms



---
## Resources:
- [Why $\mathbf{X}^T \mathbf{X}$ is always positive semi-definite](https://statisticaloddsandends.wordpress.com/2018/01/31/xtx-is-always-positive-semidefinite/)
- [Dot Products and Projections](https://math.oregonstate.edu/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/dotprod/dotprod.html)
- [Understanding what $\mathbf{w}$ is and why $y =mx + b$ <=> $\mathbf{w}^\top\mathbf{x}=0$](https://www.youtube.com/watch?v=3qzWeokRYTA)
- [Learn Differential Equations: Up Close with Gilbert Strang and Cleve Moler on MIT Opencourseware](https://ocw.mit.edu/resources/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/index.htm)
- [Freetext's Introduction to Linear Algebra](https://www.freetext.org/Introduction_to_Linear_Algebra/)
- [Xiong Fei Du's notes on LU, LDU, and Cramer's Rule](https://xiongfeidu.github.io/notes/4.09%20LU%20Decomposition%20and%20Cramer's%20Rule.pdf)
- [Systems of linear equations Wiki](https://en.wikipedia.org/wiki/System_of_linear_equations)
- [Intuition for what is meant by Overdetermined and Underdetermined systems](http://quickmathintuitions.org/intuition-for-overdetermined-and-underdetermined-systems-of-equations/)

