import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt
import tutorials


def nn_from_scratch():
    # mnist: sklearn.utils._bunch.Bunch = fetch_openml('mnist_784', version=1, data_home="./mnist_dataset", cache=True, return_X_y=True)
    # # X = feature matrix, representing the input data for machine learning model
    # X: pd.DataFrame = mnist.data
    # # y = Target variable or labels, which represent the values you want to predict
    # y: pd.Series = mnist.target

    # x, y = fetch_openml('mnist_784', version=1, data_home="./mnist_dataset", cache=True, return_X_y=True)
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    # Dividing every element by 255 is done in order to scale pixel values to the range of [0, 1].
    # Converting each element to 'float32' is necessary when working with machine learning because
    # it consumes less memory when compared to 'float64'.
    x = (x/255).astype('float32')

    # The to_categorical(y) functions performs one-hot encoding on the pandas Series 'y'.
    y = to_categorical(y)

    """
    train_test_split() = used to split the dataset into training and testing sets for
        machine learning purposes.
    test_size = param used to specify the proportion of the dataset allocated for the test set.
        So 15% is used for testing while the remaining is used for training
    random_state = sets the seed for the random number generator. Setting a value for this param
        ensures getting the same split each time, making code reproducible
    """
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

    dnn = tutorials.DeepNeuralNetwork(sizes=[784, 128, 64, 10])
    dnn.train(x_train, y_train, x_val, y_val)


def samson_zhang_nn():
    # https://youtu.be/w8yWXqWQYmU?si=7AUoI8c8kq7m0OB2
    # https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook
    # https://github.com/wehrley/Kaggle-Digit-Recognizer/blob/master/train.csv
    data_df: pd.DataFrame = pd.read_csv('samson_khang_data/train.csv')

    data: np.ndarray = np.array(data_df)
    m, n = data.shape  # m = rows, n = number of pixels or features
    np.random.shuffle(data)  # shuffle before splitting into dev and training sets

    # limit data dev to just the first 1000 samples
    # transpose so that each column is a sample instead of row (pixel(n) are rows)
    # each element of data_dev is a sample of size 1000
    data_dev = data[0:1000].T

    """
    (Pdb) data_dev = data[0:1000].T
    (Pdb) data_dev_df = data_df[0:1000].T
    (Pdb) data_dev
    array([[1, 0, 1, ..., 9, 6, 4],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)
    (Pdb) data_dev_df
              0    1    2    3    4    5    6    ...  993  994  995  996  997  998  999
    label       1    0    1    4    0    0    7  ...    0    2    2    5    9    6    4
    pixel0      0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    pixel1      0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    pixel2      0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    pixel3      0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    ...       ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
    pixel779    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    pixel780    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    pixel781    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    pixel782    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    pixel783    0    0    0    0    0    0    0  ...    0    0    0    0    0    0    0
    """

    Y_dev = data_dev[0]  # first row, gets the labels
    X_dev = data_dev[1:n]  # its 1:n because we want to skip the label row
    X_dev = X_dev / 255  # Dividing every element by 255 is done in order to scale pixel values to the range of [0, 1].

    data_train = data[1000:m].T  # use the remaining data samples for training
    Y_train = data_train[0]  # first row, gets the labels
    X_train = data_train[1:n]  # its 1:n because we want to skip the label row
    X_train = X_train / 255  # Dividing every element by 255 is done in order to scale pixel values to the range of [0, 1].
    _, m_train = X_train.shape  # _ is ignored while m_train is the number of columns

    # X_train[:, 0].shape  # : grabs all the rows, while the 0 grabs only the first column = (784,)

    def init_params():
        # np.random.rand() gives a number between 0 and 1, -0.5 makes sure its between +-0.5
        W1 = np.random.rand(10, 784) - 0.5  # 10 arrays of size 784
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2

    def ReLU(Z):
        return np.maximum(Z, 0)

    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def forward_prop(W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def ReLU_deriv(Z):
        return Z > 0

    def one_hot(Y):
        """
        Y.size = how many examples there are
        Y.max() + 1 = because we have a range from 0 - 9, we want to be 10 so we add 1 to 9
        """
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1  # for each row got to column specified by Y and set 1 to it
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2

    def get_predictions(A2):
        return np.argmax(A2, 0)

    def get_accuracy(predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(X, Y, alpha, iterations):
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(A2)
                print(get_accuracy(predictions, Y))
        return W1, b1, W2, b2

    def make_predictions(X, W1, b1, W2, b2):
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
        predictions = get_predictions(A2)
        return predictions

    def test_prediction(index, W1, b1, W2, b2):
        current_image = X_train[:, index, None]
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

    for i in range(10):
        test_prediction(i, W1, b1, W2, b2)



    breakpoint()


def main():
    # nn_from_scratch()
    # tutorials.patrick_loeber_tutorial()
    samson_zhang_nn()


if __name__ == '__main__':
    startTime = time.time()
    main()
    endTime = time.time()
    finalTime = (endTime - startTime)

    print("\nRunning Time:", "{:.2f}".format(finalTime) + " s")
