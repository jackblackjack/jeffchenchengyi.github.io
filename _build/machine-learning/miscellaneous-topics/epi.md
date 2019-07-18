---
interact_link: content/machine-learning/miscellaneous-topics/epi.ipynb
kernel_name: python3
has_widgets: false
title: 'Elements of Programming Interviews (Python)'
prev_page:
  url: /machine-learning/miscellaneous-topics/how-flask-app-works
  title: 'How to start a Flask App?'
next_page:
  url: /machine-learning/miscellaneous-topics/leetcode
  title: 'Leetcode (Python)'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Elements of Programming Interviews

## Chapters:
4. Primitive Types
- Arrays
- Strings
- Linked Lists
- Stacks and Queues
- Binary Trees
- Heaps
- Searching
- Hash Tables
- Sorting
- Binary Search Trees
- Recursion
- Dynamic Programming
- Greedy Algorithms and Invariants
- Graphs
- Parallel Computing
- Design Problems
- Language Questions
- Object-Oriented Design



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import collections
import itertools as it

```
</div>

</div>



---
# Chapter 4: Primitive Types



## Bit Manipulation



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
16 << 1 # Shifts 1 bit to left of bin(16)[2:], => 16*2

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
32
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
16 >> 1 # Shifts 1 bit to right of bin(16)[2:], => 16/2

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
8
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
~0 # ~x = (-x) - 1, the one's complement of the number 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
-1
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
bin(19)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'0b10011'
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
bin(20)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'0b10100'
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
bin(~19) 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'-0b10100'
```


</div>
</div>
</div>



### 4.1
Computing Parity of a word



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def parity(x): # O(n), n - word size
    result = 0
    while x:
        # x = 7, bin(x)[2:] = 111
        # x&1: 1&1=1, 0&1=0, 
        # result = result ^ (x & 1) = 000 ^ (111 & 001) = 000 ^ 001 = 001
        result ^= x & 1 
        x >>= 1
    return result

def faster_parity(x): # O(k), k - number of bits set to 1 in a particular word
    # x&(x-1) equals x with its lowest set bit erase
    result = 0
    while x:
        result ^= 1 # equivalent to oscillating between 0 and 1
        x &= x - 1 # Drops lowest set bit of x
    return result
        

```
</div>

</div>



---
# Chapter 8: Stacks and Queues



### 8.1
Implement Stack with Max API



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Stack:
    """Stack with Max element API, we will keep another 
       stack that contains the max element currently seen
       along with its count so that we dont need to re-search
       the maximum element if the maximum element is popped
    """
    # Tuple that contains the current max element and its count
    MaxWithCount = collections.namedtuple(typename='MaxWithCount', 
                                          field_names=('max_el', 'count'))
    
    def __init__(self):
        self._stack = []
        self._cached_max_with_count = []
        
    def empty(self):
        return len(self._stack) == 0
    
    def peek(self):
        if self.empty():
            raise IndexError('peek(): empty stack')
        return _stack[-1]
    
    def max(self):
        if self.empty():
            raise IndexError('max(): empty stack')
        return self._cached_max_with_count[-1].max_el
    
    def pop(self):
        if self.empty():
            raise IndexError('pop(): empty stack')
        if self.max() == self.peek():
            self._cached_max_with_count.count -= 1
            if self._cached_max_with_count.count == 0:
                self._cached_max_with_count.pop()
        
        return self._stack.pop()
        
    def push(self, element):
        self._stack.append(element)
        if self.empty():
            self._cached_max_with_count.append(MaxWithCount(max_el=element, 
                                                            count=1))
        else:
            if element == self.max():
                self._cached_max_with_count.count += 1
            elif element > self.max():
                self._cached_max_with_count.append(MaxWithCount(max_el=element, 
                                                            count=1))

```
</div>

</div>



---
# Chapter 9: Binary Trees

- Full BT: Every node other than leaves has 2 children
- Perfect BT: Full BT which all leaves have same depth
- Complete BT: BT that is filled in every level (except last must be filled left to right)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class TreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

```
</div>

</div>



## Traversals:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def __init__(self):
    self.stack = []
    self.result = []

# Recursive
def preorderTraversal(self, root: TreeNode) -> list:
    if root is not None:
        self.result.append(root.val)
        self.preorderTraversal(root.left)
        self.preorderTraversal(root.right)

    return self.result

# Recursive
def inorderTraversal(self, root: TreeNode) -> list:
    if root is not None:
        self.inorderTraversal(root.left)
        self.result.append(root.val)
        self.inorderTraversal(root.right)

    return self.result

# Recursive
def postorderTraversal(self, root: TreeNode) -> list:
    if root is not None:
        self.postorderTraversal(root.left)
        self.result.append(root.val)
        self.postorderTraversal(root.right)

    return self.result

```
</div>

</div>



### 9.1a
Test if a Binary Tree is height-balanced



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def is_balanced_binary_tree(root: Node): # O(h) Space - height of tree, O(n) Time
    BalancedStatusWithHeight = collections.namedtuple(typename='BalancedStatusWithHeight', 
                                                      field_names=('balanced', 'height'))
    
    # Post order Traversal
    def check_balanced(root: Node):
        # Leaves
        if root == None:
            return BalancedStatusWithHeight(balanced=True, height=0)
        # Inner nodes
        else:
            # Process Left
            left = check_balanced(root.left)
            if not left.balanced:
                return BalancedStatusWithHeight(balanced=False, height=0)
            # Process Right
            right = check_balanced(root.right)
            if not right.balanced:
                return BalancedStatusWithHeight(balanced=False, height=0)
            # Process Root
            return BalancedStatusWithHeight(balanced=abs(left.height - right.height) <= 1, height=max(left.height, right.height)+1)
            
    return check_balanced(root).balanced

```
</div>

</div>



### 9.1b
Write a program that returns the size of the largest subtree that is complete



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def is_complete(root: Node):
    pass

'https://www.geeksforgeeks.org/find-the-largest-complete-subtree-in-a-given-binary-tree/'

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
'https://www.geeksforgeeks.org/find-the-largest-complete-subtree-in-a-given-binary-tree/'
```


</div>
</div>
</div>



### 9.1C
Define a node in a binary tree to be $k$-balanced if the difference in the number of nodes in its left and right subtrees is no more than $k$. Design an algorithm that takes as input a binary tree and positive integer $k$, and retums a node in the binary tree such that the node is not k-balanced, but all of its descendants are k-balanced. For example, when applied to the binary tree in Figure 9.1 on Page 112, if $k = 3$, your algorithm should return Node $J$.



---
# Chapter 10: Heaps



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import heapq

```
</div>

</div>



### 10.1
Merge Sorted Files

This problem is motivated by the following scenario. You are given 500 files, each containing stock trade information for an S&P 500 company. Each trade is encoded by a line in the following format: 1232111, AAPL, 30, 456.12.

The first number is the time of the trade expressed as the number of milliseconds since the start of the day's trading. Lines within each file are sorted in increasing order of time. The remaining values are the stock symbol, number of shares, and price. You are to create a single file containing all the trades from the 500 files, sorted in order of increasing trade times. The individual files are of the order of 5-100 megabytes; the combined file will be of the order of five gigabytes. In the abstract, we are trying to solve the following problem.

Write a program that takes as input a set of sorted sequences and computes the union of these sequences as a sorted sequence. For example, if the input is <3,5,7>, <0,6>, and <0,6,28>, then the output is (0, 0, 3, 5, 6, 6, 7, 28).




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
x = [(3,5,7), (0,6), (0,6,28)]
list(heapq.merge(*x))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[0, 0, 3, 5, 6, 6, 7, 28]
```


</div>
</div>
</div>



---
# Chapter 13: Sorting

Python Sort: TimSort




---
# Chapter 18: Graphs



### 18.1
Search a maze



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
WHITE, BLACK = range(2)
Coordinate = collections.namedtuple('Coordinate', ('x', 'y'))

def search_maze(maze, s, e): # O(V+E), same as DFS
    # Performs DFS to find a feasible path
    def search_maze_helper(curr):
        if not (
            0 <= curr.x < len(maze) 
            and 0 <= curr.y < len(maze[curr.x]) 
            and maze[curr.x][curr.y] == WHITE):
            return False
        path.append(curr)
        maze[curr.x][curr.y] = BLACK # Turn it black after exploring
        
        if curr == e: # If we have reached destination, we're done
            return True
        
        if any(map(search_maze_helper,(
            Coordinate(curr.x-1, curr.y),
            Coordinate(curr.x+1, curr.y),
            Coordinate(curr.x, curr.y-1),
            Coordinate(curr.x, curr.y+1),
        ))):
            return True
        
        # Cannot find path, remove the entry appended 
        del path[-1]
        return False
        
    path = []
    if not search_maze_helper(s):
        return [] # No path between s and e
    return path

```
</div>

</div>



### 18.2
Paint a boolean matrix



---
## Resources
- [Python `collections.namedtuple`](https://pymotw.com/2/collections/namedtuple.html)
- [itertools tutorial](https://realpython.com/python-itertools/#what-is-itertools-and-why-should-you-use-it)

