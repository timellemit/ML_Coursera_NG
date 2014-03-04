import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    temp = np.dot(X,theta) - y
    J = 1./(2*m)*np.dot(temp.transpose(), temp)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros([num_iters,1])
    for i in xrange(num_iters):
        temp_theta = theta
        temp_theta -= alpha/m*np.dot(X.transpose(),np.dot(X,theta) - y)
        theta = temp_theta
        # we store all cost values to see whether they decrease 
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history 
