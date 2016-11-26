__author__ = 'Dominik'

import numpy as np
import sklearn.model_selection as modsel
import matplotlib.pyplot as plt



def cost(theta, X, Y):
    m = X[:,1].size
    cost = np.dot(X, theta) - Y
    cost **= 2
    cost = cost.sum() / 2 / m
    return cost

def gradient_step(theta, X, Y, alpha):
    m = X[:,1].size

    cost_not_squared = np.dot(X, theta) - Y
    sub = np.dot(X.transpose(), cost_not_squared) * alpha / m

    new_theta = theta - sub
    return new_theta

def gradient_descent(theta, x, y, alpha, steps):
    for i in range(steps):
        if i%100 == 0:
            print ("Current cost: ", cost(theta, x, y))
        theta = gradient_step(theta, x, y, alpha)
    return theta


#load and format data
data = np.genfromtxt("data.csv", delimiter='\t')
X = data[:,0]
Y = data[:,1]

N = len(X)
Y = np.array(Y)
Y = Y.reshape((N, 1))

X = np.array(X)
X = X.reshape((N, 1))

"""
show data
plt.plot(X, Y, 'r.')
plt.show()
"""

#add ones column on the left
ones1 = np.ones(N)[:, None]
X = np.append(ones1, X, axis=1)    #Nx2 matrix

initial_theta = np.zeros(2)[:, None]    #2x1 vector [b;a]

#prepare train and test sets
X_train, X_test, Y_train, Y_test = modsel.train_test_split\
    (X, Y, test_size=0.33, random_state=42)

#learning
alpha = 1e-05
theta = gradient_descent(initial_theta, X_train, Y_train, alpha, 2000)

print("Training cost: ", cost(theta, X_train, Y_train))
print("Test cost: ", cost(theta, X_test, Y_test))

print(X_test)
print(X_test[:,1])
print(np.amin(X_test[:,1]))
print(np.amax(X_test[:,1]))
