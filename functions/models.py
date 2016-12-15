from django.db import models
import numpy as np
import pandas as pd

class Poly(models.Model):
    coefs_comma_split = models.TextField()
    name = models.CharField(max_length=200)
    coeff = []

    def PolyPrint(self):
        if (len(self.coeff) == 0):
            return ('Empty function')
        poly_string = str(self.coeff[0])
        for i in range(1, len(self.coeff)):
            poly_string += '+' + str(self.coeff[i]) + 'x^' + str(i)
        return (poly_string)

    # def __init__(self, coeff, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.coeff = coeff

    # def __init__(self, coeff):
    #     self.coeff = coeff

    def __call__(self, x):
        """Evaluate the polynomial."""
        s = 0
        for i in range(len(self.coeff)):
            s += int(self.coeff[i])*x**i
        return s

    def __add__(self, other):
        """Return self + other as Polynomial object."""
        # Two cases:
        #
        # self:   X X X X X X X
        # other:  X X X
        #
        # or:
        #
        # self:   X X X X X
        # other:  X X X X X X X X

        # Start with the longest list and add in the other
        if len(self.coeff) > len(other.coeff):
            result_coeff = self.coeff[:]  # copy!
            for i in range(len(other.coeff)):
                result_coeff[i] += other.coeff[i]
        else:
            result_coeff = other.coeff[:] # copy!
            for i in range(len(self.coeff)):
                result_coeff[i] += self.coeff[i]
        return Poly(result_coeff)

#
# p = Poly([1, 2, 3])
#
# print(Poly.__call__(p, 1))
#
# print(p(1))
#
# # -1:0.1:1
# #
# def Tabulat(func, x_steps):
#     # x_steps = np.arange(0, 1, 1 / step_len)
#     values = []
#     for i in range(len(x_steps)):
#         values.append(p(x_steps[i]))
#
#     tabu_df = pd.DataFrame(columns=('x', 'val'))
#     tabu_df['x'] = x_steps
#     tabu_df['val'] = values
#     return tabu_df
#
# def save_to_CSV(df, file_path):
#     df.to_csv(file_path, index=False)
#
# # ====================================================================
#
# step_len = 10
# x_steps = np.arange(0, 1, 1 / step_len)
# p = Polynomial([2, 3, 4])
# tabu_df = Tabulat(p, x_steps)
# print(tabu_df)
#
# tabu_df.to_csv('~/Projects/ads_chm/Files/tabul_fun.csv', index=False)
#
# file_path = '~/Projects/ads_chm/Files/tabul_fun.csv'
# save_to_CSV(tabu_df, file_path)
#
#
# # =====================================================================
# def open_CSV(file_path):
#     return pd.DataFrame.from_csv(file_path)
#
# def integrate_tabu(tabu_df):
#     return 0
#
# def interpolation_coeff(tabu_df):
#     coefs = []
#     # save_to_CSV(coefs, )
#     return 0
#
#
# file_path = '~/Projects/ads_chm/Files/tabul_fun.csv'
# tabu_df = open_CSV(file_path)
#
#
# # ============= euler_method_one_dim ====================================================================
# # Eulers Method:
# # dx/dt = f(x,t)
# # x(t_0) = x_0
# # t = [t_0, t_0 + tau, .. , T]
# # x_(k+1) = x_k + tau * f(t_k, x_k)
# def euler_method_one_dim (fun, x_0, t_0, T, tau):
#     t_steps = np.arange(t_0, T, tau)
#     x = []
#     x_prev = x_0
#     t_prev = t_0
#     for i in range(len(t_steps)):
#         t_prev = t_steps[i]
#         x_next = x_prev + tau * fun(t_prev)  # fun(t, x)
#         x.append(x_next)
#         x_prev = x_next
#     tabu_df = pd.DataFrame(columns=('t', 'x'))
#     tabu_df['x'] = x
#     tabu_df['t'] = t_steps
#     return tabu_df
#
# save_to_CSV(euler_method_one_dim(p, 5, 0, 4, 0.001), '~/Projects/ads_chm/Files/euler_method_one_dim.csv')
#
#
#
