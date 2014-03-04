import numpy as np
import matplotlib.pyplot as plt
from feature_norm import *
from gradient_descent import *

print 'Loading data ...'

# Load Data
data = np.loadtxt("../data/lin_reg_multivariate.txt", delimiter=",")  
m = len(data);
X, y = data[:,[0,1]].reshape(m,2), data[:,2].reshape(m,1)
 
# Print out some data points
print 'First 10 examples from the dataset:'
print data[range(10),:]
 
# Scale features and set them to zero mean
print 'Normalizing Features ...'
 
X, mu, sigma = feature_norm(X)
# Add intercept term to X
X = np.hstack([np.ones([m,1]),X])

print 'Running gradient descent ...'
 
# Choose some alpha value
alpha = 0.01
num_iters = 400
 
# Init Theta and Run Gradient Descent 
theta = np.zeros([3, 1]);
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters);
 
# Plot the convergence graph
plt.plot(range(len(J_history)), J_history)
plt.xlabel('Number of iterations');
plt.ylabel('Cost J');
plt.show()
 
# Display gradient descent's result
print 'Theta computed from gradient descent:'
print theta
 
# Estimate the price of a 1650 sq-ft, 3 br house
pred_X = np.hstack([np.array([1]),(np.array([1650,3])-mu)/sigma])
price = np.dot(np.matrix(theta).transpose(),pred_X)

print 'Predicted price of a 1650 sq-ft, 3 br house' +\
        '(using normal equations): $%f' % price