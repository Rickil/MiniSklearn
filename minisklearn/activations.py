import numpy as np
from typing import Union

class ActivationFunctions:
    @staticmethod
    def sigmoid(x: Union[float, np.ndarray], derivative: bool = False) -> Union[float, np.ndarray]:
        if derivative:
            s = 1/(1+np.exp(-x))
            return s * (1 - s)
        else:
            return 1/(1+np.exp(-x))

    @staticmethod
    def tanh(x: Union[float, np.ndarray], derivative: bool = False) -> Union[float, np.ndarray]:
        if derivative:
            return 1-np.square(np.tanh(x))
        else:
            return np.tanh(x)

    @staticmethod
    def relu(x: Union[float, np.ndarray], derivative: bool = False) -> Union[float, np.ndarray]:
        if derivative:
            return (x > 0).astype(float)
        else:
            return np.maximum(0, x)

    @staticmethod
    def softmax(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)