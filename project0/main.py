import numpy as np


def randomization(n):
  """
  Arg:
    n - an integer
  Returns:
    A - a randomly-generated nx1 Numpy array.
  """
  A = np.random.random([n, 1])
  return A


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.random([h, w])
    B = np.random.random([h, w])
    s = np.add(A,B)
    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    x = np.add(A,B)
    s = np.linalg.norm(x)
    return s


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    w = weights.transpose()
    z = np.tanh(np.dot(w,inputs))
    return z


def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x < y:
      return x*y
    else:
      return x/y


def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    vfunc= np.vectorize(scalar_function)
    return vfunc(x, y)


#print(randomization(5))
#print(operations(2, 3))
#print(norm(2, 3))
#print(neural_network(np.random.random([2,1]), np.random.random([2,1])))
#print(scalar_function(2, 3))
print(vector_function([1, 2, 3, 4], 2))