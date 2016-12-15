# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .models import Poly

class Polynomial(object):
    def __init__(self, coefficients):
        self.coeff = coefficients

    def __call__(self, x):
        # Return polinomial value
        s = 0
        for i in range(len(self.coeff)):
            s += int(self.coeff[i])*x**i
        return s

    def PolyPrint(self):
        if (len(self.coeff) == 0):
            return ('Empty function')
        poly_string = ''
        for i in range(len(self.coeff) - 1, 0, -1):
            poly_string += str(self.coeff[i]) + 'x^' + str(i) + ' + '
        poly_string += str(self.coeff[0])

        return (poly_string)

    def __mul__(self, num):
        result_coeff = list(map(int, self.coeff[:])) # copy!
        for i in range(len(self.coeff)):
            result_coeff[i] *= num
        return Polynomial(result_coeff)


    def __add__(self, other):
        # Return self + other as Polynomial object
        # Start with the longest list and add in the other
        if len(self.coeff) > len(other.coeff):
            result_coeff = list(map(int, self.coeff[:]))  # copy!
            for i in range(len(other.coeff)):
                result_coeff[i] += int(other.coeff[i])
        else:
            result_coeff = list(map(int, other.coeff[:])) # copy!
            for i in range(len(self.coeff)):
                result_coeff[i] += int(self.coeff[i])
        return Polynomial(result_coeff)

import parser
from math import sin, cos, sqrt, log, e

class FunctionRho:

    def __init__(self, expression):
        self.formula = parser.expr(expression).compile()

    def __call__(self, *args):
        w = args[0]
        return eval(self.formula)

class FunctionSz:

    def __init__(self, expression):
        self.formula = parser.expr(expression).compile()

    def __call__(self, *args):
        t = args[0]
        return eval(self.formula)

class FunctionF:

    def __init__(self, expression, beta, z_func, s_func):
        self.formula = parser.expr(expression).compile()
        self.beta = beta
        self.z = z_func
        self.s = s_func

    def __call__(self, t, x):
        z = self.z(t)
        S = self.s(t)
        beta = self.beta
        return eval(self.formula)

class TabulatedFunction:

    def __init__(self, tabulated_values, grid):
        self.values = tabulated_values
        self.grid = grid

    def __getitem__(self, i):
        return self.values[i]

class Interpolation:

    def __init__(self, splines, grid):
        self.grid = grid
        self.splines = splines

    def __call__(self, x):
        if (x < self.grid[0]) or (x > self.grid[len(self.grid) - 1]):
            print(x)
            raise RuntimeError("x is not in domain of interpolation")
        for i in range(len(self.grid) - 1):
            if (x >= self.grid[i]) & (x <= self.grid[i + 1]):
                break
        return self.splines[i](x)

class Polinomial:
    def __init__(self, degree, coefs, x0):
        self.degree = degree
        self.coefs = coefs
        self.x0 = x0
        if degree + 1 != len(coefs):
            raise RuntimeError("Polinom is initialised incorrectly")

    def __call__(self, x):
        res = 0
        for i in range(self.degree + 1):
            res += self.coefs[i] * (x - self.x0) ** i
        return res

class Solution:

    def __init__(self, y_tab, x_tab, S_tab, x_0, y_0, beta, C1, C2, loss):
        self.y_tab = y_tab
        self.x_tab = x_tab
        self.S_tab = S_tab
        self.x_0 = x_0
        self.y_0 = y_0
        self.beta = beta
        self.C1 = C1
        self.C2 = C2
        self.loss = loss
