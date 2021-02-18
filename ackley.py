import numpy as np
import sys
from matplotlib import pyplot as plt
import math

def function(x):
  #returns the value at the given coordinate
  try:
    x1 = np.float(x[0])
    x2 = np.float(x[1])
  except:
    print("input must be 2-D numerical vector")
    sys.exit(1)
  part_1 = math.sqrt(0.5*(x1*x1 + x2*x2))
  part_2 = 0.5*(math.cos(2*np.pi*x1) + math.cos(2*np.pi*x2))
  value = math.exp(1) + 20 -20*math.exp(-0.2 * part_1) - math.exp(part_2)
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
  part_1 = math.sqrt(0.5*(x1*x1 + x2*x2))
  part_2 = 0.5*(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2))
  gradient = [0,0]
  if np.abs(part_1) > 1e-12:
    gradient = 2 / part_1 * np.exp(-0.2 * part_1) * np.asarray([x1,x2]) + math.exp(part_2) * math.pi * np.sin(2*math.pi*np.asarray([x1,x2]))
  #returning the value
  return gradient

def ackley_function_range(x_range_array):
  #returns an array of values for the given x range of values
  value = np.empty([len(x_range_array[0])])
  for i in range(len(x_range_array[0])):
    
    #returns the point value of the given coordinate
    part_1 = -0.2*math.sqrt(0.5*(x_range_array[0][i]*x_range_array[0][i] + x_range_array[1][i]*x_range_array[1][i]))
    part_2 = 0.5*(math.cos(2*np.pi*x_range_array[0][i]) + math.cos(2*np.pi*x_range_array[1][i]))
    
    value_point = math.exp(1) + 20 -20*math.exp(part_1) - math.exp(part_2)
    value[i] = value_point
  #returning the value array
  return value

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
  plt.contour(X, Y, Z, label='Ackley Function')
  
  def plot_ackley(x1_range,x2_range):
    #This would be the input for the Function
    x_range_array = [x1_range,x2_range]
    #generate the z range
    z_range = ackley_function_range(x_range_array)
    #plotting the function
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x1_range, x2_range, z_range, label='Ackley Function')
    return fig, ax