import numpy as np
from matplotlib import pyplot as plt

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    faces = faces.reshape(-1, 48, 48)
    faces = faces.T
    faces = np.reshape(faces, (faces.shape[0] ** 2, faces.shape[2]))
    ones = np.ones((faces.shape[1]))
    faces = np.vstack((faces, ones))
    return faces

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.

def worst_errors(w, Xte, yte):
    errors = abs(yte - (Xte.T.dot(w)))
    errors = np.argsort(errors)
    errors = errors[::-1]
    errors = errors[0:5]
    return errors

def fMSE (w, Xtilde, y):
    fmse = 1/5000 * (y - Xtilde.T.dot(w)).T.dot(y - Xtilde.T.dot(w))
    return fmse
# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    grand_fmse = 1/5000 * (y - Xtilde.T.dot(w)).T.dot(y - Xtilde.T.dot(w))
    return grand_fmse

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    X_trans = np.dot(Xtilde, Xtilde.T)
    Xid = np.eye(X_trans.shape[0])
    weight = np.dot(np.linalg.solve(X_trans, Xid), Xtilde.dot(y))
    return weight
    # return Xtilde.T.dot(weight)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = .1
    regularization = gradientDescent(Xtilde,y, ALPHA)
    return regularization

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    np.random.seed(2)
    w = .01 * np.random.randn((Xtilde.shape[0]))
    for i in range(T):
        gradient = 1 / T * Xtilde.dot(Xtilde.T.dot(w) - y)
        gradient[:-1] = gradient[:-1] + ((alpha/T) * w[:-1])
        w = w - EPSILON * gradient
    return w
    # yhat = Xtilde.T.dot(w)
    # return yhat

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    # Xtilde_te = np.load("age_regression_Xte.npy") #comment this out if you want to reproduce the most eggrigious errors
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")
    #
    w1 = method1(Xtilde_tr, ytr)
    fMSE_ = fMSE(w1, Xtilde_tr, ytr)
    print("One_shots_solution_training MSE: ", fMSE_)
    w2 = method2(Xtilde_tr, ytr)
    simple_gradient_desc = gradfMSE(w2, Xtilde_tr, ytr)
    print("Simple_gradient_descent_training: ", simple_gradient_desc)
    # # worst = [884, 1640, 830, 581, 939] # the worst indicies for regularized
    w3 = method3(Xtilde_tr, ytr)
    w1_test = fMSE(w1, Xtilde_te, yte)
    w2_test = gradfMSE(w2, Xtilde_te, yte)
    w3_test = gradfMSE(w3, Xtilde_te, yte)
    # errors = worst_errors(w3, Xtilde_te, yte) #uncomment this to find the worst errors
    # print(errors)
    # print(errors)
    # for error in worst:
    #     err = Xtilde_te[error, :, :]
    #     plt.imshow(err)
    #     plt.show()
    reg_gradient_desc = gradfMSE(w3, Xtilde_tr, ytr)
    print("Regularized_gradient_descent_MSE_training: ", reg_gradient_desc)
    print("One_shot_test_MSE: " + str(w1_test), "\n","Simple_gradient_descent_test_MSE: " + str(w2_test),  "Regularized_gradient_"
                                                                                                 "descent_MSE_training: "
          + str(w3_test))
    w1 = w1[:-1]
    w1 = np.reshape(w1, (48, 48))
    w2 = w2[:-1]
    w3 = w3[:-1]
    w2 = np.reshape(w2, (48, 48))
    w3 = np.reshape(w3, (48, 48))
    w2 = w2.T
    w3 = w3.T
    plt.imshow(w1)
    plt.show()
    plt.imshow(w2)
    plt.show()
    plt.imshow(w3)
    plt.show()

    # Report fMSE cost using each of the three learned weight vectors
    ...
