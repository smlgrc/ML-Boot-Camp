import numpy as np
import pandas as pd
import altair as alt
from sympy import Matrix


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
    



def main():
    # vectors()
    # matrices()
    linear_and_affine_mappings()


if __name__ == '__main__':
    main()
