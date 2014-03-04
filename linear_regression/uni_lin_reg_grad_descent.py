from gradient_descent import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # profit of a company in different cities
    data = np.loadtxt("../data/lin_reg_univariate.txt", delimiter=",")  
    
    # Number of observations
    m = len(data)
    
    # X - population of a cities, y - profits in cities 
    X, y = data[:,0].reshape(m,1), data[:,1].reshape(m,1)
    
    # plotting the data
    plt.scatter(X, y, c = 'r', marker='x', hold=True)
    plt.title('Company profit by city')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
    
    # adding a column of ones to x
    X = np.hstack([np.ones([m,1]),X.reshape(m,1)])
    # initialize fitting parameters
    theta = np.zeros([2, 1])
    # gradient descent settings
    iterations = 1500;
    alpha = 0.01;
    
    # running gradient descent
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    print 'Theta found by gradient descent: ', theta[0], theta[1] 
    
    # plotting fitted line
    plt.scatter(X[:,1], y, c = 'r', marker='x', hold=True)
    plt.plot(X[:,1], np.dot(X,theta), c = 'b')
    plt.title('Training data. Linear regression')
    plt.show()
    
    predict1, predict2 = np.dot(np.array([1, 3.5]), theta), np.dot(np.array([1, 7]), theta)
    print 'For population = 35000, we predict a profit of $%f' % (predict1*10000)
    print 'For population = 70000, we predict a profit of $%f' % (predict2*10000)
    
    # plot convergence - J(theta) by iterations
    plt.plot(range(len(J_history)), J_history)
    plt.xlabel('Number of iterations');
    plt.ylabel('Cost J');
    plt.show()





