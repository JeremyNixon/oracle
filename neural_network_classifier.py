import pandas as pd
import numpy as np
import sklearn.cross_validation
from collections import Counter
from sklearn import preprocessing 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
Y = df['quality'].values
le = preprocessing.LabelEncoder().fit(Y)
Y = le.transform(Y)
df = preprocessing.scale(df.drop('quality',1))
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(df, Y, test_size = .80, random_state=42)

class Topology:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

class Dense:
    def __init__(self, input_shape, output_shape, n_hidden):
        self.n_hidden = n_hidden
        self.weights = np.random.randn(input_shape, output_shape) * .01
        self.hidden_layer = None
        self.delta = None
        self.layer_type = 'Dense'
        
    def forward(self, forward_data):
        self.hidden_layer = np.matmul(forward_data, self.weights)
        return self.hidden_layer
    
    def prev_delta(self, delta):
        return np.matmul(delta, self.weights.T)
    
    def compute_gradient(self, backward_data, delta):
        return np.matmul(backward_data.T, delta)

class Output:
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.randn(input_shape, output_shape) * .01
        self.output_softmax = None
        self.delta = None
        self.layer_type = 'Output'
        
    def forward(self, forward_data):
        output_raw = np.matmul(forward_data, self.weights)
        self.output_softmax = np.exp(output_raw)
        self.output_softmax = self.output_softmax / np.sum(self.output_softmax, axis=1, keepdims=True)
        return self.output_softmax
    
    def prev_delta(self, delta):
        self.output_softmax[range(batch_size), y_batch] -= 1
        self.output_softmax /= batch_size
        return np.matmul(self.output_softmax, self.weights.T)
    
    def compute_gradient(self, backward_data, delta):
        return np.matmul(backward_data.T, delta)
        
    
class Activation:
    def __init__(self, kind):
        self.layer_type = 'Activation'
        self.kind = kind
        self.hidden_layer = None
        self.delta = None
        self.output_softmax = None
        
    def forward(self, input_data):
        if self.kind == 'relu':
            self.hidden_layer = np.maximum(0, input_data)
            return self.hidden_layer

        if self.kind == 'linear':
            self.hidden_layer = input_data
            return self.hidden_layer
        
        if self.kind == 'softmax':
            self.hidden_layer = np.exp(input_data)
            self.output_softmax = self.hidden_layer / np.sum(self.hidden_layer, axis=1, keepdims=True)
            return self.output_softmax
        
    def prev_delta(self, delta):
        if self.kind == 'relu':
            delta[self.hidden_layer <= 0] = 0
            return delta
        
        if self.kind == 'linear':
            return delta
        
        if self.kind == 'softmax':
            self.output_softmax /= len(self.output_softmax)
            self.output_softmax[range(batch_size), y_batch] -= 1
            self.delta = self.output_softmax
            return self.delta
        
    def compute_gradient(self, fill1, fill2):
        return
    
def evaluate2H(t, x_test):
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    x = x_test
    # Forward
    for layer in t.layers:
        x = layer.forward(x)
    return x
    
def neural_network(x_train, y_train, x_test, y_test, t, num_hidden1=100, num_hidden2=100, num_hidden3=100, lr=0.1, num_iters=10000, batch_size=32):
    # Add bias
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    
    # Get Important Shapes
    n_row, n_col = np.shape(x_train)
    n_classes = len(np.unique(y_train))
    
    # Iterate Through Backpropagation
    for iteration in xrange(num_iters):
        
        stochastic_sample = np.random.randint(0, n_row-1, batch_size)
            
        x_batch = x_train[stochastic_sample]
        y_batch = y_train[stochastic_sample]
        
        x = x_batch
        # Forward
        for layer in t.layers:
            x = layer.forward(x)
        
        output_softmax = x
        
        # Backward
        
        output_softmax[range(batch_size), y_batch] -= 1
        t.layers[-1].delta = output_softmax/batch_size
        for i in xrange(len(t.layers)-2, 0, -1):
            if t.layers[i].layer_type == 'Dense':
                t.layers[i].delta = t.layers[i].prev_delta(t.layers[i+1].delta)
            if t.layers[i].layer_type == 'Activation':
                t.layers[i].delta = t.layers[i].prev_delta(t.layers[i+1].delta)
        
        for i in xrange(len(t.layers)-2, -1, -1):
            if t.layers[i].layer_type == 'Dense':
                if i == 0:
                    t.layers[i].weights -= lr * t.layers[i].compute_gradient(x_batch, t.layers[i+1].delta)
                else:
                    t.layers[i].weights -= lr * t.layers[i].compute_gradient(t.layers[i-1].hidden_layer, t.layers[i+1].delta)

        weights = []
        for layer in t.layers:
            if layer.layer_type == 'Dense':
                weights.append(layer.weights)
    return t

t = Topology()
t.add(Dense(input_shape = 12, output_shape = 100, n_hidden = 100))
t.add(Activation('relu'))
t.add(Dense(100, 100, 100))
t.add(Activation('relu'))
t.add(Dense(100, 6, 0))
t.add(Activation('softmax'))

topology = neural_network(x_train, y_train, x_test, y_test, t, num_iters = 10000)

out = evaluate2H(topology, x_test)
print Counter(y_test-np.argmax(out, axis=1))[0]/float(len(y_test))