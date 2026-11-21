import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if(len(seqs) == 0): return np.zeros((0,0), dtype = int)
    m = 0
    if (max_len == None):
        for x in seqs:
            m = max(m,len(x))
        L = m
    else:
        L = max_len
    out = np.full((len(seqs),L),pad_value)
    for i,s in enumerate(seqs):
        k = min(len(s),L)
        out[i, :k] = s[:k]
        
    return np.array(out)
            