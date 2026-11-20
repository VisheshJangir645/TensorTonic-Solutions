import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    N,D = X.shape
    w = np.zeros(D)
    b = 0.0
    for _ in range(steps):
        p = _sigmoid(X@w+b)
        err = p-y
        grad_w = X.T @ err / N
        Bgrad_w = np.mean(err)
        w = w-lr*grad_w
        b = b-lr*Bgrad_w
    return (w,float(b))