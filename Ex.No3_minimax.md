# Ex.No: 3  Implementation of Minimax Search
### DATE:26.08.2024                                                                            
### REGISTER NUMBER : 212222040034
### AIM: 
Write a mini-max search algorithm to find the optimal value of MAX Player from the given graph.
### Algorithm:
1. Start the program
2. import the math package
3. Specify the score value of leaf nodes and find the depth of binary tree from leaf nodes.
4. Define the minimax function
5. If maximum depth is reached then get the score value of leaf node.
6. Max player find the maximum value by calling the minmax function recursively.
7. Min player find the minimum value by calling the minmax function recursively.
8. Call the minimax function  and print the optimum value of Max player.
9. Stop the program. 

### Program:
```
# Initial values of Alpha and Beta
MAX, MIN = 1000, -1000
# Returns optimal value for current player
#(Initially called for root and maximizer)
def minimax(depth, nodeIndex, maximizingPlayer,values, alpha, beta):
# Terminating condition. i.e
# leaf node is reached
    if depth == 3:
       return values[nodeIndex]
   
    if maximizingPlayer:
        best = MIN
        # Recur for left and right children
        for i in range(0, 2):
          val = minimax(depth + 1, nodeIndex * 2 + i,False, values,alpha, beta)
          best = max(best, val)
          alpha = max(alpha, best)
          # Alpha Beta Pruning
          if beta <= alpha:
             break
        return best
    else:
        best = MAX
        # Recur for left and
        # right children
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i,True, values, alpha,beta)
            best = min(best, val)
            beta = min(beta, best)
            # Alpha Beta Pruning
            if beta <= alpha:
               break
        return best
values = [3, 5, 6, 9, 1, 2, 0, -1]
print("The optimal value is :", minimax(0, 0, True, values, MIN, MAX))
```

### Output:

![image](https://github.com/user-attachments/assets/b5e77c99-0a47-454c-81fa-f7942189a5ea)


### Result:
Thus the optimum value of max player was found using minimax search.
