import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    correct_logprobs = -np.log(y_pred[range(m), y_true])
    return np.sum(correct_logprobs) / m

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
