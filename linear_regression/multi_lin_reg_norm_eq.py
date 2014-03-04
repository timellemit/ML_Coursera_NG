import numpy as np

print 'Loading data ...'

# Load Data
data = np.loadtxt("../data/lin_reg_multivariate.txt", delimiter=",")  
m = len(data);
X, y = data[:,[0,1]].reshape(m,2), data[:,2].reshape(m,1)

# Add intercept term to X
X = np.hstack([np.ones([m,1]),X])
  
print 'Solving with normal equations...'
  
# Calculate the parameters from the normal equation
print np.dot(X.transpose(),X)
print (np.linalg.inv(np.dot(X.transpose(),X))).shape
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)), X.transpose()),y)
  
# Display normal equation's result
print 'Theta computed from the normal equations:'
print theta

price = np.dot(theta.transpose(), np.array([1, 1650, 3]))

print 'Predicted price of a 1650 sq-ft, 3 br house' +\
        '(using normal equations): $%f' % price
  
