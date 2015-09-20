import pandas as pd
import numpy as np

def step_gradient(intersect_current, slope_current, x_train, y_train, learning_rate):
    intersect_gradient = 0
    slope_gradient = 0
    N = float(len(x_train))
    for i in range(len(x_train)):
        intersect_gradient += -(2/N) * (y_train[i] - ((slope_current*x_train[i] + intersect_current)))
        slope_gradient += -(2/N) * x_train[i] * (y_train[i] - ((slope_current * x_train[i] + intersect_current)))
    new_intersect = intersect_current - (learning_rate * intersect_gradient)
    new_slope = slope_current - (learning_rate * slope_gradient)
    return [new_intersect, new_slope]

def linear_regression(x_train, y_train, num_iterations = 1000, learning_rate=.0001):
	slope = -1
	intercept = 0
	for i in range(num_iterations):
	    intercept, slope = step_gradient(intercept, slope, x_train, y_train, learning_rate)
	return intercept, slope
