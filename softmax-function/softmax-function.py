import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    import numpy as np
    x = np.asarray(x)
    
    if x.ndim == 1:
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x)
    
    elif x.ndim == 2:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    else:
        raise ValueError("Input must be one- or two-dimensional")