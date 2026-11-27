import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    s_ = np.array(s)
    w_ = np.array(w)
    g_ = np.array(g)
    s_new = s_*beta+(1-beta)*(g_**2)
    w_new = w_-(lr/(s_new+eps)**(1/2))*g_
    return (w_new, s_new)