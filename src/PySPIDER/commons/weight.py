import numpy as np
from numpy import polynomial as p


def weight_leg(q):
    # return qth Legendre polynomial as a polynomial object
    orders = np.zeros(q + 1)
    orders[q] = 1
    leg = p.legendre.leg2poly(orders)
    poly = p.Polynomial(leg)
    return poly


def weight_poly(m):
    # return (x^2-1)^m as a polynomial object
    return p.Polynomial([-1, 0, 1]) ** m


def weight_1d(m, q, k, dx):
    poly = weight_leg(q) * weight_poly(m)
    poly = poly / dx**k
    return poly.deriv(k)
