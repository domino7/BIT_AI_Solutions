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
        """
        if i%100 == 0:
            print ("Current cost: ", cost(theta, x, y))
        """
        theta = gradient_step(theta, x, y, alpha)
    return theta



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

linearTestCost = cost(theta, X_test, Y_test)
print("LINEAR: Training cost: \t", cost(theta, X_train, Y_train))
print("LINEAR: Test cost: \t\t", linearTestCost)



### regression with one parameter y = ax^2 + bx + c
X_vals = X[:,1].reshape(m, 1)
X2 = np.append(X, X_vals**2, axis=1)    #(n+1)x m  matrix
initial_theta2 = np.zeros((3, 1))

X_train2, X_test2, Y_train2, Y_test2 = modsel.train_test_split\
    (X2, Y, test_size=0.33, random_state=42)

alpha2 = 1e-9
steps2 = 2000
theta2 = gradient_descent(initial_theta2, X_train2, Y_train2, alpha2, steps2)

qadraticTestCost = cost(theta2, X_test2, Y_test2)
print("QUADRATIC: Training cost: \t", cost(theta2, X_train2, Y_train2))
print("QUADRATIC: Test cost: \t\t", qadraticTestCost)

#plot data
x_p = np.linspace(min(X_test[:,1]), max(X_test[:,1]))
y_lin = [theta[0] + x*theta[1] for x in x_p]
y_quad = [theta2[0] + x*theta2[1] + x**2*theta2[2] for x in x_p]

plt.plot(X_test[:,1], Y_test, 'b.')
linearPlot = plt.plot(x_p, y_lin, 'r', label="Linear cost = " + str(linearTestCost))
quadraticPlot = plt.plot(x_p, y_quad, 'g', label="Quadratic cost = " + str(qadraticTestCost))
plt.legend(loc = 2)
plt.show()

