import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pe = np.zeros((seq_len,d_model), dtype = float)
    inp = np.arange(seq_len).reshape(-1,1)
    div = np.arange(0,d_model,2)
    div = (base ** (div / d_model)).reshape(1, -1)
    angle = inp/div
    pe[:, 0::2] = np.sin(angle)
    pe[:,1::2] = np.cos(angle[:, :d_model // 2])
    return pe