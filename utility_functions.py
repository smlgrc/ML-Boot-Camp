import numpy as np
import pandas as pd
import altair as alt
from sympy import Matrix


def testing(matrix_A, matrix_B) -> np.ndarray:
    return matrix_A @ matrix_B