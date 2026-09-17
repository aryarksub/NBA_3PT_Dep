"""NBA three-point boundary in 94-by-50-foot SportVU coordinates.

The idealized boundary itself counts as two-point territory. Tracked body
coordinates do not replace recorded scoring or shooting-foot adjudication.
"""
import numpy as np

def is_three_vec(pts):
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    hoop_x = np.where(x < 47., 5.25, 88.75)
    return ((np.abs(y-25.) > 22.+1e-9)
            | (np.hypot(x-hoop_x,y-25.) > 23.75+1e-9))

def is_three(x, y):
    return bool(is_three_vec([[x,y]])[0])
