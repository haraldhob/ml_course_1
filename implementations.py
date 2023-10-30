# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # Solve normal equation and get first result
    (a, b) = tx.T @ tx, tx.T @ y
    ws = np.linalg.lstsq(a, b, rcond=None)
    w = ws[0]
    # Calculate residuals
    res = y - tx @ w
    # Calculate the Mean Squared Error
    mse = (res.T @ res) / (2 * len(y))
    return w, mse


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    N = y.shape[0]
    loss = (-1 / N) * np.sum(y*np.log(sigmoid(tx @ w)) + (1-y)*np.log(1-sigmoid(tx @ w)))
    return loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    N = y.shape[0]
    return (1 / N) * tx.T @ (sigmoid(tx @ w)-y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    return loss, w - gamma * gradient


def logistic_regression(y, tx, initial_w=None, max_iter=10000, gamma=0.1, threshold=1e-12):
    """return the loss, gradient of the loss

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iter: int and maximum amount of iterations
        gamma: float and learning rate
        threshold: float and threshold as absolute diff to stop

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gradient, loss = logistic_regression(y, tx, w)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    (array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]]))
    """
    losses = []
    w = np.zeros((tx.shape[1], 1)) if initial_w is None else initial_w
    # Iterate over max iterations. Stop when we get convergence under threshold.
    for iter in range(max_iter):
        # get loss and w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if (iter % 10) == 0:
            print("Current iteration={%s}, loss={%s}" % (iter, loss))
        # converge check
        losses.append(loss)
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]) < threshold):
            break

    loss = calculate_loss(y, tx, w)
    print("loss={%s}" % loss)
    plt.plot(losses)
    return w, loss
