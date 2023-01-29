import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import load_data

def initialisation(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1)
    b2 = np.zeros((n2, 1))

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward_propagation(_train, parametres):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']
    
    Z1 = W1.dot(_train) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)

    activations = {
        'A1': A1,
        'A2': A2
    }

    return activations

def back_propagation(_train, y_train, parametres, activations):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']
    m = y_train.shape[1]

    dZ2 = A2 - y_train
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(_train.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    
    return gradients

def update(gradients,parametres,  learning_rate):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def predict(_train, parametres):
    activations = forward_propagation(_train, parametres)
    A2 = activations['A2']
    return A2 >= 0.5

def neural_network(x_train, y_train, n1=32, learning_rate = 0.1, n_iter = 1000, test=None):

    # initialisation parametres
    n0 = x_train.shape[0]
    n2 = y_train.shape[0]
    np.random.seed(0)
    parametres = initialisation(n0, n1, n2)

    train_loss = []
    test_acc = []
    train_acc = []
    history = []

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(x_train, parametres)
        A2 = activations['A2']

        # Plot courbe d'apprentissage
        train_loss.append(log_loss(y_train.flatten(), A2.flatten()))
        y_pred = predict(x_train, parametres)
        train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
        if test != None:
            y_test = predict(test[0], parametres)
            test_acc.append(accuracy_score(test[1].flatten(), y_test.flatten()))
        
        history.append([parametres.copy(), train_loss, train_acc, i])

        # mise a jour
        gradients = back_propagation(x_train, y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    if test != None:
        plt.plot(test_acc, label='test acc')
    plt.legend()
    plt.show()

    return parametres

    
if __name__ == '__main__':
    """
    X_train, y_train = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))
    
    print("X_train.shape = " + str(X_train.shape))
    print("y_train.shape = " + str(y_train.shape))
    
    plt.scatter(X_train[0, :], X_train[1, :], c=y_train, cmap="summer")
    plt.show()
    """
    x_train, y_train, x_test, y_test = load_data()
    plt.figure(figsize=(32, 16))
    for i in range(10):
        plt.subplot(4, 5, i+1)
        plt.imshow(x_train[i], cmap='gray')
        plt.tight_layout()
    plt.show() 
    
    y_train = y_train.T
    y_test = y_test.T
    
    x_train = x_train.T
    x_train_reshape = x_train.reshape(-1, x_train.shape[-1])/x_train.max()
    
    x_test = x_test.T
    x_test_reshape = x_test.reshape(-1, x_test.shape[-1])/x_test.max()
    
    print("x_train shape : " + str(x_train_reshape.shape))
    print("x_test shape : " + str(x_test_reshape.shape))
    
    m_train = 1000
    m_test = 200
    
    x_test_reshape = x_test_reshape[:, :m_test]
    x_train_reshape = x_train_reshape[:, :m_train]
    y_test = y_test[:, :m_test]
    y_train = y_train[:, :m_train]
    
    parameters = neural_network(x_train_reshape, y_train, n1=255, learning_rate = 0.1, n_iter = 1000, test = (x_test_reshape, y_test))
    