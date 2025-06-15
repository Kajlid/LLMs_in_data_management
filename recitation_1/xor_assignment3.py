import random
import math

def ReLU(x,derivative = False):
    if x>=0 and not derivative:
        return x
    if x>=0:
        return 1
    return 0

def step(x):
    if x <= 0:
        return 0
    
    else: 
        return 1 
    

# Zero out first hidden node (since we only need two hidden nodes for this XOR problem with two inputs)
# ReLU(a·1 + d·x1 + g·x2)  
# => a = 0, d = 0, g = 0 and for output, k = 0

# Hidden node 2 should handle (1, 0)
# ReLU(b·1 + e·x1 + h·x2)   
# -1 + 2 + 0 = 1

# Hidden node 3 should handle (0, 1)
# ReLU(c·1 + f·x1 + i·x2)
# -1 + 0 + 2 * 1 = 1

# out = step(j·1 + k·H1 + l·H2 + m·H3)
# for (0, 0)
# out = step(-0.5 + 0 + 0 + 0) = step(-0.5) = 0

# for (0, 1):
# out = step(-0.5 + 0 + 0 + 1) = step(0.5) = 1

# for (1, 0):
# out = step(-0.5 + 0 + 1 + 0) = step(0.5) = 1

# for (1, 1):
# out = step(-0.5 + 0 + 0 + 0)  = step(-0.5) = 0

weights = {
    # biases
    'a':  0,    # H1 turned off
    'b': -1,    # H2 bias
    'c': -1,    # H3 bias

    # X1
    'd':  0,    # H1 turned off
    'e':  2,    
    'f': -1,    

    # X2
    'g':  0,    # H1 turned off
    'h': -1,   
    'i':  2,   

    # output layer
    'j': -0.5,  # bias
    'k':  0,    # weight on H1 for output
    'l':  1,    # weight on H2 for output
    'm':  1     # weight on H3 for output
}

def xor(x1, x2):
    bias = 1
    H1 = ReLU(weights['a'] * bias + weights['d'] * x1 + weights['g'] * x2)
    H2 = ReLU(weights['b'] * bias + weights['e'] * x1 + weights['h'] * x2)
    H3 = ReLU(weights['c'] * bias + weights['f'] * x1 + weights['i'] * x2)
    
    out = weights['j'] * bias + weights['k'] * H1 +weights['l'] * H2 + weights['m'] * H3
    
    return step(out)

for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"({x1},{x2}) gives output: {xor(x1, x2)}")