import numpy as np

from minisklearn.activations import ActivationFunctions

class MLP:
    def __init__(self, hidden_layers=(64,), activation='relu', learning_rate=0.01, epochs=100, batch_size=32):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # One-hot encode y
        y_onehot = np.eye(n_classes)[y]
        
        # Initialize weights
        self.weights = []
        self.biases = []
        layer_sizes = [n_features] + list(self.hidden_layers) + [n_classes]
        
        for i in range(1, len(layer_sizes)):
            input_size = layer_sizes[i-1]
            output_size = layer_sizes[i]
            self.weights.append(self._initialize_weights(input_size, output_size))
            self.biases.append(np.zeros(output_size))

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            for batch_idx in range(0, n_samples, self.batch_size):
                begin, end = batch_idx, min(batch_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[begin:end]
                y_batch = y_shuffled[begin:end]
                
                # Forward pass
                layer_outputs, layer_inputs = self._forward_pass(X_batch)
                y_pred = layer_outputs[-1]
                
                # Backward pass
                self._backward_pass(X_batch, y_batch, layer_outputs, layer_inputs)

    def predict(self, X):
        layer_outputs, _ = self._forward_pass(X)
        return np.argmax(layer_outputs[-1], axis=1)

    def _initialize_weights(self, input_size, output_size):
        if self.activation == "relu":
            std = np.sqrt(2.0 / input_size)  # He initialization
        else:
            std = np.sqrt(1.0 / input_size)  # Xavier initialization
        return np.random.normal(0, std, size=(input_size, output_size))

    def _forward_pass(self, X):
        layer_inputs = []
        layer_outputs = [X]
        
        current_input = X
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            z = np.dot(current_input, w) + b
            layer_inputs.append(z)
            
            # Activation (except for output layer)
            if i < len(self.weights) - 1:
                current_input = getattr(ActivationFunctions, self.activation)(z)
            else:
                current_input = ActivationFunctions.softmax(z)
            
            layer_outputs.append(current_input)
            
        return layer_outputs, layer_inputs

    def _backward_pass(self, X, y, layer_outputs, layer_inputs):
        # Calculate gradients
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        error = layer_outputs[-1] - y
        gradients_w[-1] = np.dot(layer_outputs[-2].T, error) / X.shape[0]
        gradients_b[-1] = np.mean(error, axis=0)
        
        # Backpropagate through hidden layers
        for l in range(len(self.weights)-2, -1, -1):
            error = np.dot(error, self.weights[l+1].T) * getattr(ActivationFunctions, self.activation)(layer_inputs[l], derivative=True)
            gradients_w[l] = np.dot(layer_outputs[l].T, error) / X.shape[0]
            gradients_b[l] = np.mean(error, axis=0)
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def _compute_loss(self, y_true, y_pred):
        # Categorical cross-entropy
        return -np.mean(y_true * np.log(y_pred + 1e-15))