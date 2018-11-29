import pandas as pd
import numpy as np
import sklearn.model_selection
from collections import Counter
from sklearn import preprocessing 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
Y = df['quality'].values
le = preprocessing.LabelEncoder().fit(Y)
Y = le.transform(Y)
df = preprocessing.scale(df.drop('quality',1))
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df, Y, test_size = .80, random_state=42)

class Dense:
    def __init__(self, input_shape, output_shape, n_hidden):
        self.n_hidden = n_hidden
        self.weights = np.random.randn(input_shape, output_shape) * .1
        self.moment1 = np.zeros((input_shape, output_shape))
        self.moment2 = np.zeros((input_shape, output_shape))
        self.gradients = []
        self.deltas = []
        self.hidden_layer = None
        self.delta = None
        self.layer_type = 'Dense'
        
    def forward(self, forward_data):
        self.hidden_layer = np.matmul(forward_data, self.weights)
        return self.hidden_layer
    
    def prev_delta(self, delta):
        delta = np.matmul(delta, (self.weights).T)
        self.deltas.append(delta)
        return delta
    
    def compute_gradient(self, backward_data, delta):
        gradient = np.matmul(backward_data.T, delta)
        self.gradients.append(gradient)
        return gradient

class Activation:
    def __init__(self, kind):
        self.layer_type = 'Activation'
        self.kind = kind
        self.hidden_layer = None
        self.delta = None
        self.output_softmax = None
        self.deltas = []
        
    def forward(self, input_data):
        if self.kind == 'relu':
            self.hidden_layer = np.maximum(0, input_data)
            return self.hidden_layer
        
        if self.kind == 'linear':
            self.hidden_layer = input_data
            return self.hidden_layer
        
        if self.kind == 'softmax':
            # Deal with Overflow Problems
            input_data = input_data - np.amax(input_data, axis=1, keepdims=True)
            # Compute Softmax
            self.hidden_layer = np.exp(input_data)
            self.output_softmax = self.hidden_layer / np.sum(self.hidden_layer, axis=1, keepdims=True)
            return self.output_softmax
        
    def prev_delta(self, delta):
        if self.kind == 'relu':
            delta[self.hidden_layer <= 0] = 0
            self.deltas.append(delta)
            return delta
        
        if self.kind == 'linear':
            self.deltas.append(delta)
            return delta
        
        if self.kind == 'softmax':
            self.output_softmax /= len(self.output_softmax)
            self.output_softmax[range(batch_size), y_batch] -= 1
            self.deltas.append(output_softmax)
            self.delta = self.output_softmax
            return self.delta
        
    def compute_gradient(self, fill1, fill2):
        return
    
class Topology:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, x_test):
        x = x_test
        # Forward
        for layer in self.layers:
            x = layer.forward(x)    
        output_softmax = x
        return np.argmax(output_softmax, axis=1)
        
    def evaluate(self, x_test, y_test):
        x = x_test
        # Forward
        for layer in self.layers:
            x = layer.forward(x)    
        output_softmax = x
        accuracy = Counter(y_test-np.argmax(output_softmax, axis=1))[0]/float(len(y_test))
        return accuracy

    def fit(self, x_train, y_train, x_test, y_test, lr=0.1, s=0.9, r=0.999, num_iters=10000, batch_size=32, optimizer='adam'):
        t = self
        # Get Important Shapes
        n_row, n_col = np.shape(x_train)
        n_classes = len(np.unique(y_train))

        # Init Space for Losses
        losses = []

        # Init time step
        step = 0

        # Init numerical stability
        numerical_stability = .0000001

        # Iterate Through Backpropagation
        for iteration in xrange(num_iters):
            step += 1

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

            # Compute Errors
            for i in xrange(len(t.layers)-2, 0, -1):
                t.layers[i].delta = t.layers[i].prev_delta(t.layers[i+1].delta)

            # Compute and Update Gradients
            for i in xrange(len(t.layers)-2, -1, -1):
                if t.layers[i].layer_type == 'Dense' or t.layers[i].layer_type == 'Convolution2D':
                    if t.layers[i].layer_type == 'Dense':
                        if i == 0:
                            gradient = t.layers[i].compute_gradient(x_batch, t.layers[i+1].delta)
                        else:
                            gradient = t.layers[i].compute_gradient(t.layers[i-1].hidden_layer, t.layers[i+1].delta)

                    if t.layers[i].layer_type == 'Convolution2D':
                        if i == 0:
                            gradient = t.layers[i].compute_gradient(x_batch, t.layers[i+1].delta)
                        else:
                            gradient = t.layers[i].compute_gradient(t.layers[i-1].hidden_layer, t.layers[i+1].delta)


                    if optimizer == 'adam':

                        gradient = gradient + numerical_stability

                        # Update biased moment estimates
                        t.layers[i].moment1 = s * t.layers[i].moment1 + (1-s) * gradient
                        t.layers[i].moment2 = r * t.layers[i].moment2 + (1-r) * (gradient * gradient)

                        # Correct bias in moment estimates
                        m1_unbiased = t.layers[i].moment1/(1-s**step)
                        m2_unbiased = t.layers[i].moment2/(1-r**step)

                        # Update Layer Weights
                        t.layers[i].weights -= lr * m1_unbiased/(np.sqrt(m2_unbiased) + numerical_stability)

                    if optimizer == 'sgd':

                        # Update Layer Weights
                        t.layers[i].weights -= lr * gradient

                    if optimizer == 'momentum':

                        # Update Momentum
                        t.layers[i].moment1 = s * t.layers[i].moment1 + (1-s) * gradient

                        # Update Layer Weights
                        t.layers[i].weights -= (lr * gradient) + t.layers[i].moment1      

            if iteration % 10 == 0:
                training_loss = self.evaluate(x_train[:1000], y_train[:1000])
                validation_loss = self.evaluate(x_test[:1000], y_test[:1000])
                losses.append([training_loss, validation_loss])
            if iteration % 10 == 0:
                print "Iteration ", iteration, ": Train Loss = ", training_loss, " Val Loss = ", validation_loss

t = Topology()
t.add(Dense(input_shape = 11, output_shape = 100, n_hidden = 100))
t.add(Activation('relu'))
t.add(Dense(100, 100, 100))
t.add(Activation('relu'))
t.add(Dense(100, 6, 0))
t.add(Activation('softmax'))
t.fit(x_train, y_train, x_test, y_test, num_iters=1000)

preds = t.predict(x_test)
print Counter(y_test-preds)[0]/float(len(y_test))