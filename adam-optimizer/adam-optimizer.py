import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    m_ = np.array(m);
    v_ = np.array(v);
    param_ = np.array(param)
    grad_ = np.array(grad)
    m_new =  m_*beta1 + (1-beta1)*grad_
    v_new = v_*beta2 + (1-beta2)*(grad_**2)
    m_cor = m_new/(1-beta1**t)
    v_cor = v_new/(1-beta2**t)
    denom = v_cor**(1/2)+eps
    param_new = param_-(m_cor/denom)*lr
    return (param_new, m_new, v_new)