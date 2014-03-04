import numpy as np

def feature_norm(X):
    mu, sigma = np.mean(X, axis=0), np.std(X, axis=0)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma

if __name__ == "__main__":
    X = np.array([[1,2],[3,4]])
    print feature_norm(X)

    data = np.loadtxt("data/lin_reg_multivariate.txt", delimiter=",")  
    m = len(data);
    X, y = data[:,[0,1]].reshape(m,2), data[:,2].reshape(m,1)
    print feature_norm(X)


