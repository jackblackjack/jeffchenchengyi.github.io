---
interact_link: content/machine-learning/miscellaneous-topics/leetcode.ipynb
kernel_name: python3
has_widgets: false
title: 'Leetcode (Python)'
prev_page:
  url: /machine-learning/miscellaneous-topics/epi
  title: 'Elements of Programming Interviews (Python)'
next_page:
  url: /machine-learning/miscellaneous-topics/insurance
  title: 'Prudential Insurance'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Leetcode Questions and Solutions
Herein lies all the leetcode questions and solutions (with explanation) in Python

1. [Trees](#trees)
2. [Dynamic Programming](#dp)
3. [General](#general)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from typing import List
from collections import defaultdict

```
</div>

</div>



<a id='trees'></a>

---
## Trees



### [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution144:
    """
    Question:
    ---------
    Given a binary tree, return the preorder traversal of its nodes' values.

    Example:

    Input: [1,null,2,3]
       1
        \
         2
        /
       3

    Output: [1,2,3]
    Follow up: Recursive solution is trivial, could you do it iteratively?

    Solution:
    ---------
    recursive: process root, process left, process right
    iterative: use a stack to simulate recursion
    
    """
    def __init__(self):
        self.stack = []
        self.result = []
    
    # Recursive
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root is not None:
            self.result.append(root.val)
            self.preorderTraversal(root.left)
            self.preorderTraversal(root.right)
            
        return self.result
    
    # Iterative
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        self.stack.append(root)
        while len(self.stack) > 0:
            el = self.stack.pop()
            if el is not None:
                self.result.append(el.val)
                self.stack.append(el.right)
                self.stack.append(el.left)
                
        return self.result

```
</div>

</div>



### [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution94:
    """
    Question:
    ---------
    Given a binary tree, return the inorder traversal of its nodes' values.

    Example:

    Input: [1,null,2,3]
       1
        \
         2
        /
       3

    Output: [1,3,2]
    Follow up: Recursive solution is trivial, could you do it iteratively?
    
    Solution:
    ---------
    recursive - process left subtree, root, right subtree
    iterative - if root is not None, push it to the stack and make left child the root, 
                if root is None, we have to backtrack upwards, meaning that we can process 
                the last element pushed into the stack and make its right the root, unless 
                the stack is empty which means we are at the right bottom most node
    """
    def __init__(self):
        self.result = []
        self.stack = []
    
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        
        if root is not None:
            self.inorderTraversal(root.left)
            self.result.append(root.val)
            self.inorderTraversal(root.right)
        
        return self.result
        
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        
        while True:
            if root is not None:
                self.stack.append(root)
                root = root.left
            else:
                if len(self.stack) == 0:
                    break
                
                root = self.stack.pop()
                self.result.append(root.val)
                root = root.right
                
        return self.result

```
</div>

</div>



### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution236:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        Question:
        ---------
        Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

        According to the definition of LCA on Wikipedia: 
        “The lowest common ancestor is defined between two nodes p and q as the lowest node in T 
        that has both p and q as descendants (where we allow a node to be a descendant of itself).”

        Given the following binary tree:  root = [3,5,1,6,2,0,8,null,null,7,4]

        Example 1:

        Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
        Output: 3
        Explanation: The LCA of nodes 5 and 1 is 3.
        Example 2:

        Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
        Output: 5
        Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.

        Note:
        All of the nodes' values will be unique.
        p and q are different and both values will exist in the binary tree.
        
        Solution:
        ---------
        Recursively check if the left subtree or right subtree contains p or q.
        Node is LCA either when both left subtree and right subtree contain p / q
        OR when Node is in itself p / q and one of the subtrees contain the other node.
        """
        # Base Case 1: When root is None
        if root is None:
            return None
        
        # Base Case 2: When root is either p or q
        if root == p or root == q:
            return root
        
        # Recursive Cases
        left_subtree = self.lowestCommonAncestor(root.left, p, q)
        right_subtree = self.lowestCommonAncestor(root.right, p, q)
        
        # When i already found both nodes, return myself
        if left_subtree and right_subtree:
            return root
        # Return whichever I have found and continue going
        # up the tree until the current node contains both
        # nodes in its left and right subtree
        else:
            return left_subtree if left_subtree else right_subtree

```
</div>

</div>



### [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution102:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        """
        Question:
        ---------
        Given a binary tree, return the level order traversal of its nodes' values.
        (ie, from left to right, level by level).

        For example:
        Given binary tree [3,9,20,null,null,15,7],
            3
           / \
          9  20
            /  \
           15   7
        return its level order traversal as:
        [
          [3],
          [9,20],
          [15,7]
        ]

        Solution:
        ---------
        https://www.youtube.com/watch?v=NjdOhYKjFrU
        
        For each node, we will:
        1. Enqueue the root node
        2. Enqueue None to signal the end of the first level
        3. While our queue is not empty:
            - Dequeue the node and store it in the result list
            - Enqueue the left and right child of the node
            - if top of queue is None, it means we've finished a level and 
              we will change the level of the resulting list and enqueue None
              (This works because whenever we have finished enqueue the children
              of a level, we have reached the end of the level and we will see the None
              that is stored)
        
        """
        # Edge Case Check
        if root is None:
            return []
        
        # Initialize the list we will be using to store resulting BST traversal
        level_order_result = []
        
        # Initialize the level that we will be on
        level_idx = 0
        level_order_result.append([])
        
        # Queue we will be using 
        level_order_q = []
        
        # Enqueue the root node
        level_order_q.append(root)
        
        # Enqueue None to signal end of level
        level_order_q.append(None)
        
        # Loop through each level
        while len(level_order_q) != 0:
            curr_node = level_order_q.pop(0)
            level_order_result[level_idx].append(curr_node.val)
            if curr_node.left:
                level_order_q.append(curr_node.left)
            if curr_node.right:
                level_order_q.append(curr_node.right)
            if level_order_q[0] == None:
                level_order_q.pop(0)
                # Let's start the new level
                if len(level_order_q) != 0 and level_order_q[0] != None:
                    level_idx += 1
                    level_order_result.append([])
                    level_order_q.append(None)
                # We've traversed the whole tree
                else:
                    break
        
        return level_order_result

```
</div>

</div>



### [987. Vertical Order Traversal of a Binary Tree](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution987:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        """
        Question:
        ---------
        Given a binary tree, return the vertical order traversal of its nodes values.

        For each node at position (X, Y), its left and right children respectively will 
        be at positions (X-1, Y-1) and (X+1, Y-1).

        Running a vertical line from X = -infinity to X = +infinity, whenever the 
        vertical line touches some nodes, we report the values of the nodes in order 
        from top to bottom (decreasing Y coordinates).

        If two nodes have the same position, then the value of the node that is reported first is the value that is smaller.

        Return an list of non-empty reports in order of X coordinate.  Every report will have a list of values of nodes.
        
        Example 1:

            3
           / \
          9   20
              / \
             15  7

        Input: [3,9,20,null,null,15,7]
        Output: [[9],[3,15],[20],[7]]
        Explanation: 
        Without loss of generality, we can assume the root node is at position (0, 0):
        Then, the node with value 9 occurs at position (-1, -1);
        The nodes with values 3 and 15 occur at positions (0, 0) and (0, -2);
        The node with value 20 occurs at position (1, -1);
        The node with value 7 occurs at position (2, -2).
        
        Example 2:
        
              1
            /   \
           2     3
          / \   / \
         4   5 6   7

        Input: [1,2,3,4,5,6,7]
        Output: [[4],[2],[1,5,6],[3],[7]]
        Explanation: 
        The node with value 5 and the node with value 6 have the same position according to the given scheme.
        However, in the report "[1,5,6]", the node value of 5 comes first since 5 is smaller than 6.

        Note:
        The tree will have between 1 and 1000 nodes.
        Each node's value will be between 0 and 1000.
        
        Solution: Iterative
        -------------------
        Notice that the vertical order traversal depends on the horizontal distance of each node from
        the root.
        Let Hd be the horizontal distance and the Root node's Hd = 0. Any left child of the root 
        node is Hd = -1 and right child is Hd = +1. 
        
        We will use a queue just like in level order traversal and also a dictionary in order to track all the nodes
        that have the same horizontal distance from the root node.
        
        Solution: Recursive
        -------------------
        
        
        """
        # Initialize queue we will use
        level_order_q = []
        
        # Dictionary of List of tuples to store the 
        # key: horizontal distance away from root, 
        # value: node value and vertical distance away from root
        nodes = defaultdict(list)
        
        # Enqueue the root node and the horizontal distance and vertical
        level_order_q.append((root, 0, 0))
        
        # Go through all nodes
        while level_order_q:
            curr_node, x, y = level_order_q.pop(0)
            if curr_node.left:
                level_order_q.append((curr_node.left, x-1, y+1))
            if curr_node.right:
                level_order_q.append((curr_node.right, x+1, y+1))
            nodes[x].append((curr_node.val, y))
    
        # Have custom comparator to compare the vertical distance first and then the node values
        return [[val for val, y in sorted(nodes[x], key=lambda x: (x[1], x[0]))] for x in sorted(nodes.keys())]
        

```
</div>

</div>



<a id='dp'></a>

---
## Dynamic Programming



### [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution5:
    def longestPalindrome(self, s: str) -> str:
        """
        Question:
        ---------
        Given a string s, find the longest palindromic substring in s. 
        You may assume that the maximum length of s is 1000.

        Example 1:

        Input: "babad"
        Output: "bab"
        Note: "aba" is also a valid answer.
        Example 2:

        Input: "cbbd"
        Output: "bb"
        
        Solution: O(n^2)
        ----------------
           A         B        A        C         
        A [[(0,0)=1, (0,1)=0, (0,2)=1, (0,3)=0], 
        B  [(1,0)  , (1,1)=1, (1,2)=0, (1,3)=0],
        A  [(2,0)  , (2,1)  , (2,2)=1, (2,3)=0],
        C  [(3,0)  , (3,1)  , (3,2)  , (3,3)=1]]
         
        """
        # Initialize the table for tabulation
        table = [[False for _ in range(len(s))] for _ in range(len(s))]
        
        # Set diagonal to Trues because a single char is a palindrome
        for i in range(len(s)):
            table[i][i] = True
            
        # Set max length of palindrome found to be 1 because single char is palindrome
        max_len = 1
        
        # Initialize start index of our longest palindromic substring
        max_len_start_idx = 0
        
        # Check if 2 char substrings are palindromes (We have to keep this
        # separate from the general case of 3 or more because of we can't check 
        # whether the inner substring is a palindrome or not here)
        for start_idx in range(len(s)-2+1):
            end_idx = start_idx + 2 - 1
            if s[start_idx] == s[end_idx]:
                table[start_idx][end_idx] = True
                max_len, max_len_start_idx = 2, start_idx
        
        # We'll be traversing table topleft to bottomright diagonally
        # Outer loop is responsible for telling us which "level", AKA length of palindrome
        # we're looking for from k = 3 to n (longest possible palindrome being the entire str)
        # Next Inner Loop will go through each k length substring and check if
        # the edges of substring are equal and also if the inner substring is a palindrome
        for k in range(3, len(s)+1):
            for start_idx in range(len(s)-k+1):
                end_idx = start_idx + k - 1
                
                # Check if edges are equal, check if inner substring is palindrome
                if s[start_idx] == s[end_idx] and table[start_idx+1][end_idx-1]:
                    table[start_idx][end_idx] = True
                    if k > max_len:
                        max_len, max_len_start_idx = k, start_idx
                    
        return s[max_len_start_idx:max_len_start_idx+max_len]

```
</div>

</div>



### [516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution516:
    def longestPalindromeSubseq(self, s: str) -> int:
        """
        Question:
        ---------
        Given a string s, find the longest palindromic subsequence's length in s. 
        You may assume that the maximum length of s is 1000.

        Example 1:
        Input:

        "bbbab"
        Output:
        4
        One possible longest palindromic subsequence is "bbbb".
        Example 2:
        Input:

        "cbbd"
        Output:
        2
        One possible longest palindromic subsequence is "bb".
        
        Solution:
        ---------
        https://www.youtube.com/watch?v=_nCsPn7_OgI
        
        Similar to the Longest Palindromic Substring, we will fill the table diagonally, each diagonal representing
        the longest subsequence possible in a string of length 1, 2, ..., n. 
        
        If the edges of the substring are equal,
        the longest subsequence possible in length k 
        = 2 (for the equal edges) + longest subsequence possible in length k-2
        
        Else, the longest subsequence possible in length k 
        = MAX(LEFT: longest subsequence possible in length k-1, RIGHT: longest subsequence possible in length k-1)
        
        """
        # Initialize Table
        table = [[0 for _ in range(len(s))] for _ in range(len(s))]
        
        # Base Cases:
        # All values on diagonal = 1 because all single char substrings have longest subsequence
        # of 1
        for idx in range(len(s)):
            table[idx][idx] = 1
            
        # Longest subseqence for length k = 2 substring will either be 1 or 2
        for start_idx in range(len(s)-2+1):
            end_idx = start_idx + 2-1
            if s[start_idx] == s[end_idx]:
                table[start_idx][end_idx] = 2
            else:
                table[start_idx][end_idx] = 1
            
        # Recursive Cases:
        # Outer Loop chooses length of substring we're looking at k = 3, ..., n
        # Inner Loop changes the index
        for k in range(3, len(s)+1):
            for start_idx in range(len(s)-k+1):
                end_idx = start_idx + k-1
                if s[start_idx] == s[end_idx]:
                    table[start_idx][end_idx] = 2 + table[start_idx+1][end_idx-1]
                else:
                    table[start_idx][end_idx] = max(table[start_idx][end_idx-1], table[start_idx+1][end_idx])
                    
        # Longest Palindromic Subsequence for length n 
        return table[0][len(s)-1]

```
</div>

</div>



### [62. Unique Paths](https://leetcode.com/problems/unique-paths/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution62:
    def uniquePaths(self, m: int, n: int) -> int:
        """
        Question:
        ---------
        A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

        The robot can only move either down or right at any point in time. 
        The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

        How many possible unique paths are there?
        
        Start
         __ __ __ __ __ __ __
        |==|__|__|__|__|__|__|
        |__|__|__|__|__|__|__|
        |__|__|__|__|__|__|==|Goal
        
        Above is a 7 x 3 grid. How many possible unique paths are there?

        Note: m and n will be at most 100.

        Example 1:

        Input: m = 3, n = 2
        Output: 3
        Explanation:
        From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
        1. Right -> Right -> Down
        2. Right -> Down -> Right
        3. Down -> Right -> Right
        Example 2:

        Input: m = 7, n = 3
        Output: 28
        
        Solution:
        ---------
        https://www.youtube.com/watch?v=getrj0j1Krg
        
        All cells on top of the grid and on the left most side of the grid only have 1 unique path from 
        the start.
        
        (Optimal Substructure)
        Each inner cell of the grid has # of unique paths 
        = # of unique paths to cell above it + # of unique paths to cell on its left
        
        """
        # Initialize table for tabulation
        table = [[0 for _ in range(m)] for _ in range(n)]
        
        # Base Cases:
        # Label the top row and leftmost column with 1 
        # because there's only 1 unique path to each of those grid cells
        for idx, cell in enumerate(table[0]):
            table[0][idx] = 1
        
        for idx in range(n):
            table[idx][0] = 1
            
        # Fill table from top left to bottom right manner
        for row_idx in range(1, n):
            for col_idx in range(1, m):
                table[row_idx][col_idx] = table[row_idx-1][col_idx] + table[row_idx][col_idx-1]
                
        # Final answer to unique paths is located in bottom right corner of table
        return table[n-1][m-1]
        
        
        
        
        
        

```
</div>

</div>



### [63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution63:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """
        Question:
        ---------
        A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

        The robot can only move either down or right at any point in time. 
        The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

        Now consider if some obstacles are added to the grids. How many unique paths would there be?
        
        Start
         __ __ __ __ __ __ __
        |==|__|__|__|1 |__|__|
        |__|1 |__|__|__|__|__|
        |__|__|__|__|__|__|==|Goal
        
        An obstacle and empty space is marked as 1 and 0 respectively in the grid.

        Note: m and n will be at most 100.

        Example 1:

        Input:
        [
          [0,0,0],
          [0,1,0],
          [0,0,0]
        ]
        Output: 2
        Explanation:
        There is one obstacle in the middle of the 3x3 grid above.
        There are two ways to reach the bottom-right corner:
        1. Right -> Right -> Down -> Down
        2. Down -> Down -> Right -> Right
        
        Solution:
        ---------
        Same as Unique Paths I, just that we do some obstacle checking before and during filling up the table
        
        """
        # Initialize table for tabulation, n rows, m cols
        m, n = len(obstacleGrid[0]), len(obstacleGrid)
        table = [[0 for _ in range(m)] for _ in range(n)]

        # Base Cases:
        # Set top rows to 1 unique path unless an obstacle is encountered, then we'll just break
        # because there are no ways to get to any cell after the obstacle. Same for first column
        for col_idx in range(m):
            if obstacleGrid[0][col_idx] == 1:
                break
            else:
                table[0][col_idx] = 1
                
        for row_idx in range(n):
            if obstacleGrid[row_idx][0] == 1:
                break
            else:
                table[row_idx][0] = 1
                
        # Fill table top left to bottom right, checking if there's an obstacle
        for row_idx in range(1, n):
            for col_idx in range(1, m):
                if obstacleGrid[row_idx][col_idx] == 0:
                    table[row_idx][col_idx] = table[row_idx-1][col_idx] + table[row_idx][col_idx-1]
                    
        # The answer to total number of unique paths with obstacles is in bottom right cell
        return table[n-1][m-1]

```
</div>

</div>



### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution96:
    def numTrees(self, n: int) -> int:
        """
        Question:
        ---------
        Given n, how many structurally unique BST's (binary search trees) that store values 1 ... n?

        Example:

        Input: 3
        Output: 5
        Explanation:
        Given n = 3, there are a total of 5 unique BST's:

           1         3     3      2      1
            \       /     /      / \      \
             3     2     1      1   3      2
            /     /       \                 \
           2     1         2                 3
        
        Solution:
        ---------
        Resource: https://www.youtube.com/watch?v=GgP75HAvrlY
        
        Create list of 1 to n.
        
        Loop through from root of bst = 1 to n. Because of the BST property, we know that left subtree
        values will always be smaller than right subtree values -> So as we're going through the list of
        values, everything to the left of our current value (root value) will be in our left subtree and
        everything to the right will be in our right subtree. 
        
        Combinatorics review:
        # of BST possible with current root 
        = # of BST possibles with nodes in left subtree * # of BST possibles with nodes in right subtree
        
        We can store the # of BST possible in any subtree in a table and use dynamic programming to look
        it up to build up the table.
        
        The number sequence we create from this is also known as the Catalan numbers
        """
        # Initialize table used for tabulation
        table = [0 for _ in range(n+1)]
        
        # Base Cases:
        # # of BSTs with 0 nodes
        table[0] = 1
        
        # # of BSTs with 1 node
        table[1] = 1
        
        # Fill up table from left to right
        # Each cell's value = # of BSTs possible for each number of nodes
        # Outer loop chooses the number of nodes in our list
        # Inner loop chooses the value of the root 
        for i in range(2, n+1):
            num_BSTs = 0
            for root_val in range(1, i+1):
                num_nodes_left_subtree = root_val - 1
                num_nodes_right_subtree = i - root_val
                num_BSTs += table[num_nodes_left_subtree] * table[num_nodes_right_subtree]
                
            table[i] = num_BSTs
            
        # Total number of BSTs possible
        # that store values 1 ... n will be in the right most cell of table
        return table[n]

```
</div>

</div>



### [120. Triangle](https://leetcode.com/problems/triangle/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        """
        Question:
        ---------
        Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

        For example, given the following triangle

        [
             [2],
            [3,4],
           [6,5,7],
          [4,1,8,3]
        ]
        The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

        Note:

        Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.
        
        Solution:
        ---------
           3
          7 4
         2 4 6
        8 5 9 3

        Step 1 :
        3 0 0 0
        7 4 0 0
        2 4 6 0
        8 5 9 3

        Step 2 :
        3  0  0  0
        7  4  0  0
        10 13 15 0

        Step 3 :
        3  0  0  0
        20 19 0  0

        Step 4:
        23 0 0 0

        output : 23
        """
        if triangle != None:
            # Step 1: Pad triangle to become a lower triangular matrix
            for level in range(len(triangle)):
                triangle[level] += [0 for _ in range(len(triangle)-level-1)]
                
            # Step 2: From 2nd lowest level, calculate max(value@(level,i), value@(level, i+1))
            for level in range(len(triangle)-1, 0, -1):
                for i in range(level):
                    triangle[level-1][i] += min(triangle[level][i], triangle[level][i+1])
                    
            return triangle[0][0]
        else:
            return 0

```
</div>

</div>



### [139. Word Break](https://leetcode.com/problems/word-break/submissions/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution139:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        Question:
        ---------
        Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, 
        determine if s can be segmented into a space-separated sequence of one or more dictionary words.

        Note:

        The same word in the dictionary may be reused multiple times in the segmentation.
        You may assume the dictionary does not contain duplicate words.
        Example 1:

        Input: s = "leetcode", wordDict = ["leet", "code"]
        Output: true
        Explanation: Return true because "leetcode" can be segmented as "leet code".
        Example 2:

        Input: s = "applepenapple", wordDict = ["apple", "pen"]
        Output: true
        Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
                     Note that you are allowed to reuse a dictionary word.
        Example 3:

        Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
        Output: false
        
        Solution:
        ---------
        https://www.youtube.com/watch?v=RPeTFTKwjps
        
        We will build our substrings from the left "a" -> "ap" -> "app" and check whether we can find break
        points to segment the substring with the word dictionary. We can use dynamic programming because 
        there exists overlapping subproblems in that once we've found that we can break "applepen"... at index
        5, and ..."apple" at index 0, we can use these two to say that "applepenapple" can be segmented at 
        index 8 because the words to left of index 8 can be perfectly segmented and words on right of index 8
        can be perfectly segmented.
        """
        # Initialize table: table[i] stores whether substring of s from index 0 to index i is perfectly segmentable or not 
        table = [False for _ in range(len(s)+1)]
        
        # Base Case:
        # substring of s[:0] is perfectly segmentable
        table[0] = True

        # Recursive Case:
        # Outer loop chooses the length of substring we're looking at
        # Inner loop chooses the break point to check if the words can be perfectly segmented
        for k in range(1, len(s)+1):
            for break_idx in range(k):
                if table[break_idx] and s[break_idx:k] in wordDict:
                    table[k] = True
                    break              
                        
        return table[len(s)] 

```
</div>

</div>



<a id='general'></a>

---
## General



### [477. Total Hamming Distance](https://leetcode.com/problems/total-hamming-distance/)



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Solution477:
    def totalHammingDistance(self, nums: List[int]) -> int:
        """
        Question:
        ---------
        The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

        Now your job is to find the total Hamming distance between all pairs of the given numbers.

        Example:
        Input: 4, 14, 2

        Output: 6

        Explanation: In binary representation, the 4 is 0100, 14 is 1110, and 2 is 0010 (just
        showing the four bits relevant in this case). So the answer will be:
        HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.
        Note:
        Elements of the given array are in the range of 0 to 10^9
        Length of the array will not exceed 10^4.
        
        Solution: O(n)
        --------------
        Total hamming distance is the sum of the total hamming distance for each of the i-th bits separately.

        So, let's consider the i-th column, which consists of numbers chosen from {0, 1}. 
        The total hamming distance would be the number of pairs of numbers that are different. That is,
        
        Total hamming distance for the i-th bit =
        (the number of zeros in the i-th position) *
        (the number of ones in the i-th position).
        
        Combinatorics Review:
        Total number of pairs of (0,1) (Meaning an increment of Hamming Dist) 
        = # zeros choose 1 * # ones choose 1
        = # zeros * # ones
        
        We then add all of these together to get our answer.
        """
        # log2(10^9) < 32 bits
        bits = [[0,0] for _ in range(32)]
        
        # Go through each number in nums and
        # increment the xmod2th 0 or 1 depending
        # if x is divisible by 2 or not.
        # At the end, we 
        for x in nums:
            for bit in bits:
                bit[int(x%2)] += 1
                x /= 2
        return sum(x*y for x,y in bits)

```
</div>

</div>

