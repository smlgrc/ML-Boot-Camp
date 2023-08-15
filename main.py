import numpy as np
import pandas as pd
import altair as alt


def vectors():
    alt.themes.enable('dark')
    x = y = np.array([[1], [2], [3]])
    print(x.shape)  # (3 dimensions, 1 element on each)
    print(f'A 3 dimensional vector:\n{x}')

    # print(x + y)  # x + y = np.add(x, y)

    alpha, beta = 2, 3
    x, y = np.array([[2], [3]]), np.array([[4], [5]])
    print()
    print(alpha * x + beta * y)

    x, y = np.array([[-2], [2]]), np.array([[4], [-3]])
    print()
    print(f'\nx.transpose() @ y = \n{x.transpose() @ y}')
    print(f'\nx.transpose() * y = \n{x.transpose() * y}')
    print(f'\nx.transpose = {x.transpose()}')
    print(y)
    print(x * y)

    # TODO: Continue at VECTOR NULL SPACE - NOW THAT WE KNOW WHAT SUBSPACES AND LINEAR ...



def main():
    vectors()


if __name__ == '__main__':
    main()
