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

def y_plotLine(x, theta):
    return theta[0] + x*theta[1]


#load and format data
data = np.genfromtxt("data.csv", delimiter='\t')
X = data[:,0]
Y = data[:,1]
m = len(X)
Y = np.array(Y)
Y = Y.reshape((m, 1))
X = np.array(X)
X = X.reshape((m, 1))
#add ones column on the left
ones1 = np.ones((m, 1))
X = np.append(ones1, X, axis=1)    #m x (n+1) (80x2)  matrix


### regression with one parameter y = ax + b

initial_theta = np.zeros((2, 1))    #nx1 vector
#prepare train and test sets
X_train, X_test, Y_train, Y_test = modsel.train_test_split\
    (X, Y, test_size=0.33, random_state=42)

alpha = 1e-05
steps = 2000
theta = gradient_descent(initial_theta, X_train, Y_train, alpha, steps)

print("Training cost: ", cost(theta, X_train, Y_train))
print("Test cost: ", cost(theta, X_test, Y_test))

#plot data
plt.plot(X_test[:,1], Y_test, 'r.')
plt.plot(X_test[:,1], y_plotLine(X_test[:,1], theta))
plt.show()


