import numpy as np
import sys
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def function(x):
  #returns the value at the given coordinate
  try:
    x1 = np.float(x[0])
    x2 = np.float(x[1])
  except:
    print("input must be 2-D numerical vector")
    sys.exit(1)
  value = x1**4+x2**4
  #returning the value
  return value

def gradient(x):
    #returns the gradient at the given coordinate
    try:
        x1 = np.float(x[0])
        x2 = np.float(x[1])
    except:
        print("input must be 2-D numerical vector")
        sys.exit(1)
    gradient = [4*x1**3, 4*x2**3]
    #returning the value
    return gradient

def plot_general(fig):
  #this function will plot a general ackley function just to view it.
  limit = 1000 #number of points
  #common lower and upper limits for both x1 and x2 are used
  lower_limit = -5
  upper_limit = 5
  #generating x1 and x2 values
  x = np.linspace(-5, 5, 100)
  y = np.linspace(-5, 5, 100)
  X, Y = np.meshgrid(x, y)
  Z = np.zeros((len(X),len(Y)))
  for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = function([X[i,j],Y[i,j]])
  #plotting the function
  plt.contour(X, Y, Z, label='Simple Function')
