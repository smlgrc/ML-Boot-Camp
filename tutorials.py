# Imports
import os
import numpy as np
import pandas as pd
import altair as alt
from sympy import Matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn import utils
from random import random
from copy import deepcopy
from math import exp, log, sqrt
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List

import regression


def vectors():
    alt.themes.enable('dark')
    x = y = np.array([[1], [2], [3]])
    print(x.shape)  # (3 dimensions, 1 element on each)
    print(f'A 3 dimensional vector:\n{x}')

    # print(x + y)  # x + y = np.add(x, y)

    alpha, beta = 2, 3
    x, y = np.array([[2], [3]]), np.array([[4], [5]])
    print()
    print(f'alpha * x + beta * y =\n{alpha * x + beta * y}')

    """ Basic Vector Operations """
    x, y = np.array([[-2], [2]]), np.array([[4], [-3]])
    print(f'\nx.transpose() @ y = {x.transpose() @ y}')  # dot product
    print(f'\nx.transpose = {x.transpose()}')

    """ Vector Norms """
    # In Numpy, we can compute the L_2 (Euclidean) norm as: (pythagorean triple)
    x = np.array([[3], [4]])  # = 5.0
    print(f'\nnp.linalg.norm(x, 2) = {np.linalg.norm(x, 2)}')

    # In Numpy, we can compute the L_1 (Manhattan) norm as: (adding up absolute values of vectors)
    x = np.array([[3], [-4]])  # = 7.0
    print(f'\nnp.linalg.norm(x, 1) = {np.linalg.norm(x, 1)}')

    # In Numpy, we can comput the L_infinity (Max norm) norm as: (finding the max absolute value of vectors)
    x = np.array([[3], [-4]])
    print(f'\nnp.linalg.norm(x, np.inf) = {np.linalg.norm(x, np.inf)}')

    """ Vector inner product, length, and distance """
    # Every dot product is an inner product, but not inner product is a dot product

    # In machine learning, unless made explicit, we can safely assume that an inner
    # product refers to the dot product. We already reviewed how to compute the dot
    # product in Numpy:
    x, y = np.array([[-2], [2]]), np.array([[4], [-3]])
    print(f'\nx.T @ y = {x.T @ y}')  # -14

    # As with the inner product, usually, we can safely assume that distance stands
    # for the Euclidean distance L2 norm unless otherwise noted. To compute the L2
    # distance between a pair of vectors:
    distance = np.linalg.norm(x - y, 2)
    print(f'\nL_2 distance : {distance}')  # 7.810249675906656

    """ Vector Angles and Orthogonality """
    # In Numpy, we can compute the cos(theta) betwen a pair of vectors as:
    x, y = np.array([[1], [2]]), np.array([[5], [7]])
    # here we translate the cos(theta) definition
    cos_theta = (x.T @ y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))
    print(f'\ncos(theta) = {np.round(cos_theta, 3)}')
    # cos_theta can be used to find inverse of cosine function:
    cos_inverse = np.arccos(cos_theta)
    print(f'\ncos_inverse = {np.round(cos_inverse, 3)}')

    # We say that a pair of vectors x and y are orthogonal if their inner
    # product is zero
    x = np.array([[2], [0]])
    y = np.array([[0], [2]])

    cos_theta = (x.T @ y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))
    print(f'\ncos of the angle = {np.round(cos_theta, 3)}')

    """ Systems of Linear Equations """
    # The purpose of linear algebra as a tool is to solve systems of linear equations
    df = pd.DataFrame({"x1": [0, 2], "y1": [8, 3], "x2": [0.5, 2], "y2": [0, 3]})
    equation1 = alt.Chart(df).mark_line().encode(x="x1", y="y1")
    equation2 = alt.Chart(df).mark_line(color="red").encode(x="x2", y="y2")

    combined_chart: alt.vegalite.v5.api.LayerChart = equation1 + equation2
    combined_chart.save("chart.html")


def matrices():
    # In Numpy, we construct matrices with the array method:
    A = np.array([[0, 2],  # 1st row
                  [1, 4]])  # 2nd row
    print(f'\nA 2x2 Matrix:\n{A}')

    """ Basic Matrix Operations """
    ''' Matrix-Matrix addition '''
    A = np.array([[0, 2],
                  [1, 4]])
    B = np.array([[3, 1],
                  [-3, 2]])
    print(f'\nnp.add(A, B) = \n{np.add(A, B)}')

    ''' Matrix-Scalar Multiplication '''
    alpha = 2
    A = np.array([[1, 2],
                  [3, 4]])
    print(f'\nnp.multiply(alpha, A) = \n{np.multiply(alpha, A)}')

    ''' Matrix-Vector Multiplication: Dot Product '''
    A = np.array([[0, 2],
                  [1, 4]])
    x = np.array([[1],
                  [2]])
    print(f'\nnp.dot(A, x) = \n{np.dot(A, x)}')  # np.dot(A, x) = A @ x

    ''' Matrix-Matrix Multiplication '''
    A = np.array([[0, 2],
                  [1, 4]])
    B = np.array([[1, 3],
                  [2, 1]])
    print(f'\nnp.dot(A, B) = \n{np.dot(A, B)}')  # np.dot(A, B) = A @ B

    ''' Matrix Inverse '''
    # Not all matrices have an inverse. When Inverse A exist, we say A is
    # nonsingular or invertible, otherwise, we say its noninvertible or singular
    A = np.array([[1, 2, 1],
                  [4, 4, 5],
                  [6, 7, 7]])
    A_i = np.linalg.inv(A)
    print(f'\nA inverse:\n{A_i}')
    # Verify A inverse is correct by multiplying to original A
    I = np.round(A_i @ A)
    print(f'\nA_i times A resulsts in I_3:\n{I}')

    ''' Matrix Transpose '''
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    print(f'\nA.T =\n{A.T}')

    ''' Hadamard Product '''
    # Matrix-matrix multiplication IS NOT element-wise operation, as
    # multiplying each overlapping element of A and B. Such an operation is
    # called Hadamard Product.

    # In Numpy, we compute the Hadamard product with the * operator or multiply method:
    A = np.array([[0, 2],
                  [1, 4]])
    B = np.array([[1, 3],
                  [2, 1]])
    print(f'\nnp.multiply(A, B) =\n{np.multiply(A, B)}')  # np.multiply(A, B) = A * B

    """ Solving Systems of Linear Equations with Matrices """
    # In Numpy, we can solve a system of equations with Gaussian Elimination with the
    # linalg.solve method as:
    A = np.array([[1, 3, 5],
                  [2, 2, -1],
                  [1, 3, 2]])
    y = np.array([[-1],
                  [1],
                  [2]])
    print(f'\nnp.linalg.solve(A, y) = \n{np.linalg.solve(A, y)}')

    # NumPy does not have a method to obtain the row echelon form of a matrix.
    # But, we can use Sympy, a Python library for symbolic mathematics that counts
    # with a module for Matrices operations. SymPy has a method to obtain the reduced
    # row echelon form and the pivots, rref.
    A = Matrix([[1, 0, 1],
                [0, 1, 1]])

    A_rref, A_pivots = A.rref()
    print(f'\nReduced row echelon form of A (basis columns):\n{A_rref}')
    print(f'\nColumn pivots of A: {A_pivots}. Rank(A) = {len(A_pivots)}')

    B = Matrix([[1, 2, 3, -1],
                [2, -1, -4, 8],
                [-1, 1, 3, -5],
                [-1, 2, 5, -6],
                [-1, -2, -3, 1]])
    B_rref, B_pivots = B.rref()
    print(f'\nReduced row echelon form of B (basis columns):\n{B_rref}')
    print(f'\nColumn pivots of B: {B_pivots}. Rank(B) = {len(B_pivots)}')

    """ Matrix Norm """
    # As with vectors, we can measure teh size of a matrix by computing its norm.
    ''' Frobenius Norm '''
    # Square each entry of A, add them together, and then take the square root
    # In "NumPy", we can compute the Frobenius norm as with the" linal.norm" method and
    # "fro" as the argument:

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(f"\nnp.linalg.norm(A, 'fro') = {np.linalg.norm(A, 'fro')}")

    ''' Max Norm '''
    # The max norm or infinity norm of a matrix equals to the largest sum of the absolute
    # value of row vectors. This equals to go row by row, adding the absolute value of each
    # entry and then selecting the largest sum.
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(f"\nnp.linalg.norm(A, np.inf) = {np.linalg.norm(A, np.inf)}")

    ''' Spectral Form'''
    # The spectral norm of a matrix equals to the largest singular value sigma1
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(f"\nnp.linalg.norm(A, 2) = {np.linalg.norm(A, 2)}")


def linear_and_affine_mappings():
    alt.themes.enable('dark')
    # In general, dot products are linear mappings.
    """ Examples of Linear Mappings """
    ''' Negation Matrix '''
    # A negation matrix returns the opposite sign of each element of a vector.
    x = np.array([[-1],
                  [0],
                  [1]])
    y = np.array([[-3],
                  [0],
                  [2]])
    T = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])
    left_side_1 = T @ (x + y)
    right_side_1 = (T @ x) + (T @ y)
    print("\n===== Negation Matrix =====\nT(x+y)=T(x)+T(y)")
    print(f"Left side of the equation:\n{left_side_1}")
    print(f"Right side of the equation:\n{right_side_1}")

    alpha = 2
    left_side_2 = T @ (alpha * x)
    right_side_2 = alpha * (T @ x)
    print("\nT(αx)=αT(x)")
    print(f"Left side of the equation:\n{left_side_2}")
    print(f"Right side of the equation:\n{right_side_2}")

    ''' Reversal Matrix '''
    # A reversal matrix returns reverses the order of the elements of a vector.
    # T in this situation is the mirror Identity matrix which is used for reversing.
    x = np.array([[-1],
                  [0],
                  [1]])
    y = np.array([[-3],
                  [0],
                  [2]])
    T = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])
    x_reversal = T @ x
    y_reversal = T @ y
    left_side_1 = T @ (x + y)
    right_side_1 = (T @ x) + (T @ y)
    print("\n===== Reversal Matrix =====\nT(x+y)=T(x)+T(y)")
    print(f"x before reversal:\n{x}\nx after reversal \n{x_reversal}")
    print(f"y before reversal:\n{y}\ny after reversal \n{y_reversal}")
    print(f"Left side of the equation (add reversed vectors):\n{left_side_1}")
    print(f"Right side of the equation (add reversed vectors):\n{right_side_1}")

    alpha = 2
    left_side_2 = T @ (alpha * x)
    right_side_2 = alpha * (T @ x)
    print("\nT(αx)=αT(x), ∀α")
    print(f"Left side of the equation:\n{left_side_2}")
    print(f"Right side of the equation:\n{right_side_2}")

    """ Examples of Non-Linear Mappings """
    # ALL norms are NOT linear transformation
    # Translation is a geometric transformation that moves every vector in a vector
    # space by the same distance in a given direction. Translation is NOT a linear mapping because
    # T(x+y)=T(x)+T(y) does not hold.

    ''' Affine Mappings '''
    # The simplest way to describe affine mappings (or transformations) is a
    # linear mapping + translation. -> M(x)=Ax+b


def codebasics_tutorial():
    insurance_data_path: str = os.path.join(r"codebasics_tutorial/insurance_data.csv")
    df = pd.read_csv(insurance_data_path)
    plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)
    model = LogisticRegression()
    model.fit(X_train.values, y_train)

    print(f'\nX_test:\n{X_test.values}')
    print(f'\nmodel.predict(X_test) = {model.predict(X_test.values)}')
    print(f'\nmodel.score(X_test, y_test) = {model.score(X_test.values, y_test)}')
    print(f'\nmodel.predict_proba(X_test) =\n["prob of 0 (not buy)", "prob of 1 (buy)"]\n{model.predict_proba(X_test.values)}')

    print()
    for i in range(20, 50):
        print(f'{i} = {model.predict([[i]])}')

    print()


def assemblyAI_tutorial():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    class assemblyAI_Logistic_Regression():
        def __init__(self, lr=0.001, n_iters=1000):
            self.lr = lr
            self.n_iters = n_iters
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0

            for _ in range(self.n_iters):
                linear_pred = np.dot(X, self.weights) + self.bias
                predictions = sigmoid(linear_pred)  # wrapper -> sigmoid(X1*w1 + ... + Xnwn + b)
                dw = (1/n_samples) * np.dot(X.T, (predictions - y))
                db = (1/n_samples) * np.sum(predictions - y)

                self.weights = self.weights - self.lr*dw
                self.bias = self.bias - self.lr*db

        def predict(self, X):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)
            class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
            return class_pred

    bc: utils._bunch.Bunch = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = assemblyAI_Logistic_Regression(lr=0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    def accuracy(y_pred, y_test):
        # only counts those results that match
        return np.sum(y_pred == y_test) / (len(y_test))

    acc = accuracy(y_pred, y_test)
    print(acc)


def philipp_muens_tutorial():
    # The benefit of using Logistic Regression vs. Linear Regression because the y value
    # can be a binary value. With Logistic Regression we can map any resulting y value
    # no matter its magnitude to a value between 0 and 1.

    # At the heart of Logistic Regression is the Sigmoid Function. This function takes any
    # x value and maps it to a y value which ranges from 0 to 1.
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))
    assert sigmoid(0) == 0.5
    assert sigmoid(5) == 0.9933071490757153

    beta = [1, 2, 3, 4]
    x = [5, 6, 7, 8]

    # Function which calculates the dot product
    # See: https://en.wikipedia.org/wiki/Dot_product
    def dot(a: List[float], b: List[float]) -> float:
        assert len(a) == len(b)
        return sum([a_i * b_i for a_i, b_i in zip(a, b)])
    assert dot(beta, x) == 70

    # the squish function takes x and beta values, uses the dot function then
    # passes this result into the sigmoid function to map it to a value between 0 and 1
    # Function which turns vectors of `beta` and `x` values into a value between 0 and 1
    def squish(beta: List[float], x: List[float]) -> float:
        assert len(beta) == len(x)  # needs to be true for dot product
        return sigmoid(dot(beta, x))
    assert squish(beta, x) == 1.0

    # The negative log likelihood function which we'll use to calculate how "off" our prediction is
    def neg_log_likelihood(y: float, y_pred: float) -> float:
        return -((y * log(y_pred)) + ((1 - y) * log(1 - y_pred)))
    assert 2.30 < neg_log_likelihood(1, 0.1) < 2.31
    assert 2.30 < neg_log_likelihood(0, 0.9) < 2.31
    assert 0.10 < neg_log_likelihood(1, 0.9) < 0.11
    assert 0.10 < neg_log_likelihood(0, 0.1) < 0.11

    xs_nll: List[float] = [x / 10000 for x in range(1, 10000)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xs_nll, [neg_log_likelihood(1, x) for x in xs_nll])
    ax1.set_title('y = 1')
    ax2.plot(xs_nll, [neg_log_likelihood(0, x) for x in xs_nll])
    ax2.set_title('y = 0')
    plt.show()

    def error(ys: List[float], ys_pred: List[float]) -> float:
        assert len(ys) == len(ys_pred)
        num_items: int = len(ys)
        sum_nll: float = sum([neg_log_likelihood(y, y_pred) for y, y_pred in zip(ys, ys_pred)])
        return (1 / num_items) * sum_nll
    assert 2.30 < error([1], [0.1]) < 2.31
    assert 2.30 < error([0], [0.9]) < 2.31
    assert 0.10 < error([1], [0.9]) < 0.11
    assert 0.10 < error([0], [0.1]) < 0.11

    # Create the Python path pointing to the `marks.txt` file
    data_dir: Path = Path('logistic_regression_tutorial')
    marks_data_path: Path = data_dir / 'marks.txt'

    # Turn the data into a list of x vectors (one for every pair of x items) and a vector containing all the y items
    xs: List[List[float]] = []
    ys: List[float] = []
    with open(marks_data_path) as file:
        for line in file:
            data_point: List[str] = line.strip().split(',')
            x1: float = float(data_point[0])
            x2: float = float(data_point[1])
            y: int = int(data_point[2])
            xs.append([x1, x2])
            ys.append(y)

    # Create a scatter plot with the x1 values on the x axis and the x2 values on the y axis
    # The color is determined by the corresponding y value
    x1s: List[float] = [x[0] for x in xs]
    x2s: List[float] = [x[1] for x in xs]

    plt.scatter(x1s, x2s, c=ys)
    plt.axis([min(x1s), max(x1s), min(x2s), max(x2s)])

    plt.show()

    # Prepend a constant of 1 to every x value so that we can use the dot product later on
    for x in xs:
        x.insert(0, 1)

    print(xs[:5])

    # Rescales the data so that each item has a mean of 0 and a standard deviation of 1
    # See: https://en.wikipedia.org/wiki/Standard_score
    def z_score(data: List[List[float]]) -> List[List[float]]:
        def mean(data: List[float]) -> float:
            return sum(data) / len(data)

        def standard_deviation_sample(data: List[float]) -> float:
            num_items: int = len(data)
            mu: float = mean(data)
            return sqrt(1 / (num_items - 1) * sum([(item - mu) ** 2 for item in data]))

        data_copy: List[List[float]] = deepcopy(data)
        data_transposed = list(zip(*data_copy))
        mus: List[float] = []
        stds: List[float] = []
        for item in data_transposed:
            mus.append(mean(list(item)))
            stds.append(standard_deviation_sample(list(item)))

        for item in data_copy:
            mu: float = mean(item)
            std: float = standard_deviation_sample(item)
            for i, elem in enumerate(item):
                if stds[i] > 0.0:
                    item[i] = (elem - mus[i]) / stds[i]

        return data_copy

    xs = z_score(xs)

    print(xs[:5])

    xs_sigmoid: List[float] = [x for x in range(-10, 10)]
    ys_sigmoid: List[float] = [sigmoid(x) for x in xs_sigmoid]

    plt.plot(xs_sigmoid, ys_sigmoid)
    plt.show()

    # Find the best separation to classify the data points
    beta: List[float] = [random() / 10 for _ in range(3)]

    print(f'Starting with "beta": {beta}')

    epochs: int = 5000
    learning_rate: float = 0.01

    for epoch in range(epochs):
        # Calculate the "predictions" (squishified dot product of `beta` and `x`) based on our current `beta` vector
        ys_pred: List[float] = [squish(beta, x) for x in xs]

        # Calculate and print the error
        if epoch % 1000 == True:
            loss: float = error(ys, ys_pred)
            print(f'Epoch {epoch} --> loss: {loss}')

        # Calculate the gradient
        grad: List[float] = [0 for _ in range(len(beta))]
        for x, y in zip(xs, ys):
            err: float = squish(beta, x) - y
            for i, x_i in enumerate(x):
                grad[i] += (err * x_i)
        grad = [1 / len(x) * g_i for g_i in grad]

        # Take a small step in the direction of greatest decrease
        beta = [b + (gb * -learning_rate) for b, gb in zip(beta, grad)]

    print(f'Best estimate for "beta": {beta}')

    # Compute some statistics to see how our model is doing
    total: int = len(ys)
    thresh: float = 0.5
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    for i, x in enumerate(xs):
        y: float = ys[i]
        pred: float = squish(beta, x)
        y_pred: int = 1
        if pred < thresh:
            y_pred = 0
        if y == 1 and y_pred == 1:
            true_positives += 1
        elif y == 0 and y_pred == 0:
            true_negatives += 1
        elif y == 1 and y_pred == 0:
            false_negatives += 1
        elif y == 0 and y_pred == 1:
            false_positives += 1

    print(f'True Positives: {true_positives}')
    print(f'True Negatives: {true_negatives}')
    print(f'False Positives: {false_positives}')
    print(f'False Negatives: {false_negatives}')
    print(f'Accuracy: {(true_positives + true_negatives) / total}')
    print(f'Error rate: {(false_positives + false_negatives) / total}')

    # Plot the decision boundary
    x1s: List[float] = [x[1] for x in xs]
    x2s: List[float] = [x[2] for x in xs]
    plt.scatter(x1s, x2s, c=ys)
    plt.axis([min(x1s), max(x1s), min(x2s), max(x2s)]);

    m: float = -(beta[1] / beta[2])
    b: float = -(beta[0] / beta[2])

    x2s: List[float] = [m * x[1] + b for x in xs]

    plt.plot(x1s, x2s, '--')
    plt.show()


def patrick_loeber_tutorial():
    # https://www.youtube.com/watch?v=4swNt7PiamQ&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=2

    # Linear Regression
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = regression.LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    accu = regression.r2_score(y_test, predictions)
    print("Linear reg Accuracy:", accu)

    # Logistic reg
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # breakpoint()
    regressor = regression.LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("Logistic reg classification accuracy:", regression.accuracy(y_test, predictions))
