# Imports:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

error_viz = []

np.random.seed(2)

def split_test_train(X, y, rate=0.2):
    # First shuffle randomly 
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[p], y[p]
    
    # Split into test and train set
    i_test = round(len(X) * 0.2)
    X_test, X_train = X_shuffled[:i_test], X_shuffled[i_test:]
    y_test, y_train = y_shuffled[:i_test], y_shuffled[i_test:]
    
    return X_train, y_train, X_test, y_test

X = np.load('breast-cancer-classifier/data/X-data.npy')
y = np.load('breast-cancer-classifier/data/y-data.npy')
X_train, y_train, X_test, y_test = split_test_train(X, y)

class ReLU():
    def activate(self, x): 
        return np.maximum(x, 0.0)
    
    def derivative(self, x):
        return (x > 0) * 1  # * 1 to return a number.
    
class Sigmoid():
    def activate(self, x):
        result = np.mean(1 / (1 + np.exp(-x)))
        if math.isnan(result):
            print('here')
        return result
    
    def derivative(self, x):
        s = self.activate(x)
        if math.isnan(np.mean(s * (1 - s))):
            print('here')
        return s * (1 - s)
    
class CrossEntropyLoss():
    def calculate_loss(self, y_true, y_pred):
        result = 0
        print('y_pred', y_pred)
        if y_pred == 1.0: 
            print('here')
        if y_true == 1.0:
            result = -np.log(y_pred)
        else:
            result = -np.log(1-y_pred)
        if math.isnan(result):
            print('here')
        return result
    
    def derivative(self, y_true, y_pred): 
        r = (y_pred - y_true) / y_pred * (1 - y_pred)
        if math.isnan(r):
            print('here')
        return r
    
class MSELoss():
    def calculate_loss(self, y_true, y_pred):
        return np.mean(1/2 * (y_true - y_pred)**2)
    
    def derivative(self, y_true, y_pred):
        return y_pred - y_true  # (y_true - y_pred) * (-1)
    
# Dense (fully connected) Layer Class:
class Dense(): 
    def __init__(self, input_size, output_size, activation_function='relu', name='unnamed'):
        self.name = name
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.outputs = None
        
        if activation_function == 'relu':
            self.activation = ReLU()
        elif activation_function == 'sigmoid':
            self.activation = Sigmoid()
        else:
            self.activation = ReLU()  # Default to ReLU activation function.
        
    def print_weights(self):
        print('Weights:\n', pd.DataFrame(self.weights))
        
    def print_biases(self):
        print('Biases:\n', pd.DataFrame(self.biases))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation.activate(np.dot(self.weights, inputs) + self.biases)
        return self.outputs
    
    def backward(self, delta_l, prev_weights, x, learning_rate):                
        delta_l = np.dot(delta_l.T, prev_weights) * self.activation.derivative(self.outputs).T
        prev_weights = self.weights
        gradient = delta_l * np.dot(x, np.ones((1, len(self.outputs))))
        self.weights -= learning_rate * gradient.T 
        return delta_l, prev_weights
    
class Network:
    def __init__(self, layers, loss_function='cross_entropy'):
        self.layers = layers
        self.output = None
        
        if loss_function == 'cross_entropy':
            self.loss = CrossEntropyLoss()
        elif loss_function == 'mse':
            self.loss = MSELoss()
        else:
            self.loss = CrossEntropyLoss()  # Default to cross entropy loss. 
    
    def train(self, X_train, y_train, number_epochs, learning_rate=0.01):
        for epoch in range(number_epochs):
            error_v = 0
            
            for x, y in zip(X_train, y_train):
                # Process the forward pass. This goes through every layer.
                x = x.reshape(21, 1)  # Create a matrix for dot product in forward()
                y_hat = self.predict(x) 
                
                # Calculate the error after the forward pass. 
                loss = self.loss.calculate_loss(y, y_hat)
                    
                error = self.loss.derivative(y, y_hat)
                if math.isnan(error):
                    print('here')
                error_v += loss  # Add to the error visualisation.
                
                l0 = self.layers[0]
                l1 = self.layers[-1]
                
                # The output layer error.
                delta_l = np.multiply(error, l1.activation.derivative(l1.outputs))
                if math.isnan(np.mean(np.multiply(error, l1.activation.derivative(l1.outputs)))):
                    print('here')
                prev_weights = l1.weights
                l1.weights -= learning_rate * (l0.outputs.T * delta_l)
                
                for layer in reversed(self.layers[:-1]):
                    delta_l, prev_weights = layer.backward(delta_l, prev_weights, x, learning_rate)
            
            error_v /= len(X)
            error_viz.append(error_v)
                
                   
    def predict(self, x):
        outputs = x
        for layer in self.layers:
            outputs = layer.forward(outputs)
        self.output = outputs
        return outputs
            
            
number_inputs = X_train.shape[1]
epochs = 100

layers = [
    Dense(number_inputs, 5, activation_function='relu', name='Layer 1'),
    Dense(5, 1, activation_function='sigmoid', name='Layer 2')
]

network = Network(layers, loss_function='cross_entropy')
network.train(X, y, number_epochs=epochs)
# print_layer_outputs(network)

plt.plot(error_viz)
plt.title("Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

predictions = []
for i in X:
    predictions.append(network.predict(i.reshape(21, 1)))
    
res = pd.DataFrame()
res["predictions"] = predictions
res["actual"] = y

# res["predictions"] = res["predictions"].apply(lambda x: x[0][0])
res["predictions"] = res["predictions"].apply(lambda x: 0 if x < 0.5 else 1)

print("Accuracy:", sum(x == y for x, y in zip(res['predictions'], res['actual'])) / len(X_train) * 100)