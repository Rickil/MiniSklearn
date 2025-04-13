import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate : float = 0.01, n_iters: int = 1000):  
        # Initialize weights (coefficients) and hyperparameters  
        self.weights = None
        self.bias = 0
        self.n_features = 0 # Features size
        self.n_samples = 0 # Samples size
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:  
        # X: (n_samples, n_features)
        # Optimize weights using gradient descent to minimize log loss  
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        loss = 0
        for _ in range(self.n_iters):
            y_pred, z = self.predict(X, return_z=True)
            loss = self._compute_loss(y, z)
            grad_w, grad_b = self._compute_gradients(X, y, y_pred)

            self.weights -= self.learning_rate*grad_w
            self.bias -= self.learning_rate*grad_b
        
        predictions = self.predict(X)
        accuracy = np.sum(predictions==y)/self.n_samples
        
        return loss, accuracy

    def predict(self, X: np.ndarray, return_z: bool = False) -> np.ndarray:  
        # Return binary predictions (0 or 1) using learned weights and sigmoid  
        z = np.dot(X, self.weights) + self.bias
        z = np.clip(z, -500, 500)
        y_pred = self._sigmoid(z)
        predictions = np.where(y_pred < 0.5, 0, 1)
        if return_z:
            return predictions, z
        else:
            return predictions

    def _sigmoid(self, z: np.ndarray) -> float:  
        # Helper: Compute sigmoid of z  
        return 1/(1+np.exp(-z))

    def _compute_loss(self, y_true: np.ndarray, z: np.ndarray) -> float:  
        # Helper: Compute log loss (cross-entropy)  
        # log_losses = y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
        log_losses = y_true*np.logaddexp(0, -z) + (1-y_true)*np.logaddexp(0, z)
        average_log_loss = np.mean(log_losses)
        return average_log_loss
    
    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        error = y_pred-y_true
        grad_w = np.dot(X.T, error) / self.n_samples
        grad_b = np.mean(error)

        return grad_w, grad_b