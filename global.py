import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from data import load_data

class DeepNeuralNetwork:
    def __init__(self, X, y, hidden_layers = (16, 16, 16), threshold = 0.5):
        self.hidden_layers = hidden_layers
        self.threshold = threshold
        self.dimensions = list(hidden_layers)
        self.dimensions.insert(0, X.shape[0])
        self.dimensions.append(y.shape[0])
        self.initialisation()

    def initialisation(self):        
        self.parametres = {}
        C = len(self.dimensions)
        for c in range(1, C):
            self.parametres['W' + str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c - 1])
            self.parametres['b' + str(c)] = np.random.randn(self.dimensions[c], 1)

    def forward_propagation(self, X):
        activations = {'A0': X}
        C = len(self.parametres) // 2
        for c in range(1, C + 1):
            Z = self.parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + self.parametres['b' + str(c)]
            activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
        return activations

    def back_propagation(self, y, activations):
        m = y.shape[1]
        C = len(self.parametres) // 2

        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(self.parametres['W' + str(c)].T, dZ) *activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

        return gradients

    def update(self, gradients, learning_rate):
        C = len(self.parametres) // 2
        for c in range(1, C + 1):
            self.parametres['W' + str(c)] = self.parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
            self.parametres['b' + str(c)] = self.parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    def predict(self, X):
        activations = self.forward_propagation(X)
        C = len(self.parametres) // 2
        Af = activations['A' + str(C)]
        return Af >= self.threshold
    
    def training(self, X_train, y_train, nb_iter=1000, learning_rate = 0.001, test = None):
        C = len(self.parametres) // 2
        training_history = np.zeros((int(nb_iter), 3))
        for i in tqdm(range(nb_iter)):
            activations = self.forward_propagation(X_train)
            gradients = self.back_propagation(y_train, activations)
            self.update(gradients, learning_rate)
            if i%10 == 0:
                #pour afficher le logloss et l'accuracy à chaque 10 itérations
                Af = activations['A' + str(C)]
                training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))
                y_pred = self.predict(X_train)
                training_history[i, 1] = (accuracy_score(y_train.flatten(), y_pred.flatten()))
                if test != None:
                    y_test_pred = self.predict(test[0])
                    training_history[i, 2] = (accuracy_score(test[1].flatten(), y_test_pred.flatten()))
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(training_history[:, 0], label='train loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(training_history[:, 1], label='train acc')
        plt.legend()
        if test != None:
            plt.plot(training_history[:, 2], label='test acc')
            plt.legend()
        plt.show()
        
        return training_history

    def test(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test.flatten(), y_pred.flatten())

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    y_train = y_train.T
    y_test = y_test.T
    
    x_train = x_train.T
    x_train_reshape = x_train.reshape(-1, x_train.shape[-1])/x_train.max()
    
    x_test = x_test.T
    x_test_reshape = x_test.reshape(-1, x_test.shape[-1])/x_test.max()
    
    m_train = 1000
    m_test = 200
    
    x_test_reshape = x_test_reshape[:, :m_test]
    x_train_reshape = x_train_reshape[:, :m_train]
    y_test = y_test[:, :m_test]
    y_train = y_train[:, :m_train]
    
    network = DeepNeuralNetwork(x_train_reshape, y_train, hidden_layers = (16,16))
    network.training(x_train_reshape, y_train, nb_iter=3000, learning_rate = 0.1, test=(x_test_reshape, y_test))
    print("test accuracy : " + str(network.test(x_test_reshape, y_test)))
    

