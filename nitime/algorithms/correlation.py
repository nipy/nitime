import numpy as np

__all__ = ["seed_corrcoef"]


def seed_corrcoef(seed, target):
    """Compute seed-based correlation coefficient"""

    x = target - np.mean(target, -1)[..., np.newaxis]
    y = seed - np.mean(seed)
    xx = np.sum(x ** 2, -1)
    yy = np.sum(y ** 2, -1)
    xy = np.dot(x, y)
    r = xy / np.sqrt(xx * yy)

    return r
