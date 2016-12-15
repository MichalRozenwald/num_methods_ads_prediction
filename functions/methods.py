# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, log, e
import errno


from functions.structures import *


def Tabulate(func, x_steps):
    # x_steps = np.arange(0, 1, step_len)
    values = []
    for i in range(len(x_steps)):
        values.append(func(x_steps[i]))

    tabu_df = pd.DataFrame(columns=('x', 'val'))
    tabu_df['x'] = x_steps
    tabu_df['val'] = values
    return tabu_df

def save_to_CSV(df, file_path):
    df.to_csv(file_path, index=False)


def isBADInput(str):
    return any(c.isalpha() or (c in './!@#$%^&*()_+-=') for c in str) or (",," in str)




# ====================================================================================================================================
# ============================================= Tabulate Integral ====================================================================
# ====================================================================================================================================


def open_CSV(file_path):
    return pd.DataFrame.from_csv(file_path)

def tabulate_integral(tabu_df):
    return 'tabulate_integral function RESULTS'

def interpolation_coefficients(tabu_df):
    coefs = []
    # save_to_CSV(coefs, )
    return 'interpolation_coefficients function RESULTS + save to file'  # return the coesfs of the polinome



# ====================================================================================================================================
# ================================================   Cauchy_Form    ==================================================================
# ====================================================================================================================================
#

# ============= euler_method_one_dim ====================================================================
# Метод Эйлера:
# dx/dt = f(x,t)
# x(t_0) = x_0
# t = [t_0, t_0 + tau, .. , T]
# x_(k+1) = x_k + tau * f(t_k, x_k)
def euler_method_one_dim (fun, x_0, t_0, T, tau):
    t_steps = np.arange(t_0, T, tau)
    x = []
    x_prev = x_0
    t_prev = t_0
    for i in range(len(t_steps)):
        t_prev = t_steps[i]
        x_next = x_prev + tau * fun(t_prev)  # fun(t, x)
        x.append(x_next)
        x_prev = x_next
    tabu_df = pd.DataFrame(columns=('t', 'x'))
    tabu_df['x'] = x
    tabu_df['t'] = t_steps
    return tabu_df

# save_to_CSV(euler_method_one_dim(p, 5, 0, 4, 0.001), '~/Projects/ads_chm/Files/euler_method_one_dim.csv')


# fun_x, x_0, fun_y = z' * U(t), fun_x = (считаем) - позставляю функцию z (линейно) и b  - считаем
def euler_method_tow_dim (fun_x, x_0, fun_y, y_0, t_0, T, tau):
    t_steps = np.arange(t_0, T, tau)
    x = []
    y = []
    x_prev = x_0
    y_prev =y_0
    t_prev = t_0
    for i in range(len(t_steps)):
        t_prev = t_steps[i]
        x_next = x_prev + tau * fun_x(x_prev) # x(t_prev) # fun(t_pred, x_pred)
        y_next = y_prev + tau * fun_y(t_prev) # x(t_prev) # fun(t_pred, x_pred)
        x.append(x_next)
        y.append(y_next)
        x_prev = x_next
        y_prev = y_next
    tabu_df = pd.DataFrame(columns=('t', 'x', 'y'))
    tabu_df['t'] = t_steps
    tabu_df['x'] = x
    tabu_df['y'] = y

    return tabu_df


# Cauchy_Form:
# x_0 = forms.CharField(label='x_0', max_length=1000)
# y_0 = forms.CharField(label='y_0', max_length=1000)
# betta = forms.CharField(label='betta', max_length=1000)
# T = forms.CharField(label='T', max_length=1000)
# U = forms.CharField(label='U', max_length=1000)
# S = forms.CharField(label='S', max_length=1000)
# z = forms.CharField(label='z', max_length=1000)
# f = forms.CharField(label='f', max_length=1000)
# save_dir_path = str(form.cleaned_data['save_dir_path'])





# ====================================================================================================================================
# ============================================= Solver ODE ====================================================================
# ====================================================================================================================================

def verify(rho_a, rho_b, rho_c, s_a, z_a, err):
    err_rho = False
    if float(rho_a) / 3 + float(rho_b) / 2 + float(rho_c) != 1:
        err_rho = True
    err_sz = False
    if s_a > z_a:
        err_sz = True
    if err_rho or err_sz:
        err.append("Должны выполняться свойства:")
    return err, err_rho

def verify_rho(rho, err):
    err_rho = False
    if 1 != round(integrate(func=rho, nnodes=100, interval=[0, 1]), 1):
        err_rho = True
        err.append("Должны выполняться свойства:")
    return err, err_rho



def kramer(A, F):
    det = np.linalg.det(A)
    if not det:
        raise RuntimeError("Determinant equals to 0.")
    roots = []
    for x in range(A.shape[1]):
        tmp = A.copy()
        tmp[:, x] = F
        roots.append(float(np.linalg.det(tmp)) / det)
    return roots

def tabulate(function, grid):
    values = np.array([function(x) for x in grid])
    return TabulatedFunction(values, grid)


def interpolate(tabulated_function):
    grid = tabulated_function.grid
    N = len(grid) - 2
    matrix = np.zeros((N, N))
    values = np.zeros(N)
    for i in range(1, N+1):

        h_0 = grid[i] - grid[i-1]
        h_1 = grid[i+1] - grid[i]

        if i != 1:
            matrix[i-1, i-2] = h_0
        if i != N:
            matrix[i-1, i] = h_1

        matrix[i-1, i-1] = 2*(h_0 + h_1)
        delta_1 = float(tabulated_function[i+1] - tabulated_function[i])
        delta_0 = float(tabulated_function[i] - tabulated_function[i-1])
        values[i-1] = 6 * (delta_1 / h_1 - delta_0 / h_0)
    derivatives_2 = kramer(matrix, values)

    splines = []
    for i in range(N+1):
        der2_0 = float(derivatives_2[i-1] if i != 0 else 0)
        der2_1 = float(derivatives_2[i] if i != N else 0)

        h = grid[i+1] - grid[i]
        der3 = (der2_1 - der2_0) / h

        f0 = float(tabulated_function[i])
        f1 = float(tabulated_function[i+1])

        der1 = (f1 - f0) / h - der2_1 * h / 6 - der2_0 * h / 3

        p = Polinomial(3, [f0, der1, der2_0 / 2, der3 / 6], grid[i])
        splines.append(p)
    return Interpolation(splines, tabulated_function.grid)


def integrate(func, nnodes, interval):
    xnet = np.linspace(interval[0], interval[1], nnodes)
    I = 0
    for i in range(nnodes - 1):
        I += (float(xnet[i + 1] - xnet[i]) / 6) * float(func(xnet[i]) + 4 * func(0.5 * (xnet[i] + xnet[i + 1])) + func(xnet[i + 1]))
    return I


def derivative(tabulated_function):
    grid = tabulated_function.grid[1:-1]
    values = tabulated_function.values
    derivative_values = np.zeros(len(grid))
    step = grid[1] - grid[0]
    for i in range(len(values)-2):
        derivative_values[i] = (values[i+2] - values[i]) / (2 * step)
    return TabulatedFunction(derivative_values, grid)


def derivative_interpolation(interpolation):
    derivative_polinomials = []
    for pol in interpolation.splines:
        new_coefs = []
        for i in range(1, len(pol.coefs)):
            new_coefs.append(i * pol.coefs[i])
        derivative_polinomials.append(Polinomial(pol.degree-1, new_coefs, pol.x0))
    new_interpolation = Interpolation(derivative_polinomials, interpolation.grid)
    return new_interpolation


def diff_equastions(func1, func2, x_0, y_0, grid):
    x = [x_0,]
    y = [y_0,]
    step = grid[1] - grid[0]
    for i in range(1, len(grid)):
        x_old = x[i - 1]
        y_old = y[i - 1]
        t_old = grid[i - 1]
        x_new = x_old + step * func1(t_old, y_old)
        y_new = y_old + step * func2(t_old, x_old)
        if y_new < 0 or y_new > 1:
            raise RuntimeError("Y must be in [0, 1]. Specify another beta.")
        x.append(x_new)
        y.append(y_new)
    return TabulatedFunction(np.array(x), grid), TabulatedFunction(np.array(y), grid)


def C1_score(x_interpol, y_interpol, rho, nnodes, tgrid, T):
    integrand_dw = lambda w: w*rho(w)
    intergral_dw = lambda t: integrate(func=integrand_dw, nnodes=nnodes, interval=[y_interpol(t), 1])
    integral_tab = tabulate(intergral_dw, grid=tgrid)
    intergral_interpol = interpolate(tabulated_function=integral_tab)
    x_deriv_interpol = derivative_interpolation(x_interpol)
    integrand_dt = lambda t: x_deriv_interpol(t) * intergral_dw(t)
    integral_dt = integrate(func=integrand_dt, nnodes=nnodes, interval=[0, T])
    C1 = 1 - integral_dt / (x_interpol(T) - x_interpol(0))
    return C1


def C2_score(x_interpol, S, T):
    return abs(x_interpol(T) - S(T)) / S(T)


def Loss(C1_score, C2_score):
    return C1_score + 10*C2_score

def build_init_graph(rho, S, z, T):
    tgrid = np.linspace(0, T, 100)
    wgrid = np.linspace(0, 1, 100)
    plt.figure(figsize=(8,6))
    plt.plot(tgrid, [rho(w) for w in wgrid], label=r"$\rho(\omega)$", linewidth=2)
    plt.legend(prop={'size':20}, loc=2)
    plt.savefig('static/outputs/results/rho_graph.png')
    plt.close()
    plt.figure(figsize=(8,6))
    plt.plot(tgrid, [S(t) for t in tgrid], label=r"$S(t)$", linewidth=2)
    plt.legend(prop={'size':20}, loc=2)
    plt.savefig('static/outputs/results/S_graph.png')
    plt.close()
    plt.figure(figsize=(8,6))
    plt.plot(tgrid, [z(t) for t in tgrid], label=r"$z(t)$", linewidth=2)
    plt.legend(prop={'size':20}, loc=2)
    plt.savefig('static/outputs/results/z_graph.png')
    plt.close()

def build_res_graphs(solutions):
    fig, axis = plt.subplots(ncols=2, nrows=len(solutions)) # , figsize=(15,5*len(solutions))
    if len(solutions) == 1:
        sol = solutions[0]
        axis[0].plot(sol.y_tab.grid, sol.y_tab.values, linewidth=2)
        # axis[0].legend(prop={'size': 20}, loc=2)
        axis[0].set_title("y(t)", fontsize=14)
        axis[0].set_xlabel("t", fontsize=14)
        axis[0].set_ylabel("y(t)", fontsize=14)

        # coefs = np.polyfit(sol.x_tab.values, sol.S_tab.values, deg=1)
        axis[1].plot(sol.x_tab.values, sol.S_tab.values, linewidth=2)
        # axis[1].plot(sol.x_tab.values, [coefs[0] * x + coefs[1] for x in sol.x_tab.values], linewidth=2)
        axis[1].set_title("S(x)", fontsize=14)
        axis[1].set_xlabel("x(t)", fontsize=14)
        axis[1].set_ylabel("S(x)", fontsize=14)
        #
        # axis[0].plot(sol.S_tab.grid, sol.S_tab.values, linewidth=2)
        # # axis[0].legend(prop={'size': 20}, loc=2)
        # axis[0].set_title("s(t)", fontsize=14)
        # axis[0].set_xlabel("t", fontsize=14)
        # axis[0].set_ylabel("y(t)", fontsize=14)
        #
        # axis[1].plot(sol.x_tab.grid, sol.x_tab.values, linewidth=2)
        # # axis[0].legend(prop={'size': 20}, loc=2)
        # axis[1].set_title("x(t)", fontsize=14)
        # axis[1].set_xlabel("t", fontsize=14)
        # axis[1].set_ylabel("x(t)", fontsize=14)
        #


        # axis[2].text(0.5, 0.5, "y(0) = " + str(sol.y_0) + "\n\nx(0) = " + str(sol.x_0) + "\n\nbeta = " + str(sol.beta) +
        #              "\n\nC1(beta) = " + str(round(sol.C1, 3)) + "\n\nC2(beta) = " + str(round(sol.C2, 3)) +
        #              "\n\nLoss(C1, C2) = " + str(round(sol.loss, 3)),
        #              size=16, ha='center', va='center')
    else:
        fig, axis = plt.subplots(ncols=2, nrows=len(solutions), figsize=(15,5*len(solutions)))  # , figsize=(15,5*len(solutions))
        for i in range(len(solutions)):
            sol = solutions[i]
            axis[i][0].plot(sol.y_tab.grid, sol.y_tab.values,  label=r"$y(t)$",  linewidth=2)
            # axis[i][0].set_title("y(t)", fontsize=14)
            axis[i][0].set_xlabel("t\n", fontsize=14)
            axis[i][0].set_ylabel("y(t)\n", fontsize=14)
            axis[i][0].legend(prop={'size': 20}, loc=2)

            coefs = np.polyfit(sol.x_tab.values, sol.S_tab.values, deg=1)
            axis[i][1].plot(sol.x_tab.values, sol.S_tab.values, label=r"$S(x)$", linewidth=2)
            # axis[i][1].plot(sol.x_tab.values, [coefs[0] * x + coefs[1] for x in sol.x_tab.values],  linewidth=2)# ,  label=r"$S(x)$"
            # axis[i][1].set_title("S(x)", fontsize=14)
            axis[i][1].set_xlabel("x(t)\n", fontsize=14)
            axis[i][1].set_ylabel("S(t)\n", fontsize=14)
            axis[i][1].legend(prop={'size': 20}, loc=2)

            title = "y_0=" + str(sol.y_0) + ", x_0=" + str(sol.x_0) + ", beta=" + str(sol.beta) +\
                    ", C1=" + str(round(sol.C1, 3)) + ", C2=" + str(round(sol.C2, 3)) +\
                    ", Loss=" + str(round(sol.loss, 3)) + "\n"
            axis[i][0].set_title(title, fontsize=16)

            # axis[i][2].text(0.5, 0.5, "y(0) = " + str(sol.y_0) + "\n\nx(0) = " + str(sol.x_0) + "\n\nbeta = " + str(sol.beta) +
            #          "\n\nC1(beta) = " + str(round(sol.C1, 3)) + "\n\nC2(beta) = " + str(round(sol.C2, 3)) +
            #          "\n\nLoss(C1, C2) = " + str(round(sol.loss, 3)),
            #          size=16, ha='center', va='center')
    fig.savefig('static/outputs/results/solution.png')


def write_to_file(solutions):
    f = open('static/outputs/results/result.txt', 'w')
    res = ''
    for sol in solutions:
        res += '\nbeta = ' + str(sol.beta)
        res += '\nC1 = ' + str(sol.C1)
        res += '\nC2 = ' + str(sol.C2)
        res += '\nloss = ' + str(sol.loss)
        res += '\nx_0 = ' + str(sol.x_0)
        res += '\ny_0 = ' + str(sol.y_0)
        res += '\ny(t):' + str(list(zip(sol.y_tab.grid, sol.y_tab.values)))
        res += '\nx(t):' + str(list(zip(sol.x_tab.grid, sol.x_tab.values)))
    f.write(res)
    f.close()


def solver(is_automatic, rho, z, S, f, y_0, x_0, T):
    build_init_graph(rho, S, z, T)
    nnodes = 100
    tgrid = np.linspace(0, T, nnodes)
    ygrid = np.linspace(0, 1, nnodes)
    S_tab = tabulate(function=S, grid=tgrid)
    z_tab = tabulate(function=z, grid=tgrid)
    z_interpol = interpolate(tabulated_function=z_tab)
    z_derivative_interpol = derivative_interpolation(interpolation=z_interpol)
    integral = lambda y: integrate(func=rho, nnodes=nnodes, interval=[y, 1])
    integral_tab = tabulate(function=integral, grid=ygrid)
    integral_interpol = interpolate(tabulated_function=integral_tab)
    func1 = lambda t, y: z_derivative_interpol(t) * integral_interpol(y)
    log = []
    if is_automatic:
        scores = []
        for f_func in f:
            try:
                x, y = diff_equastions(func1, f_func, x_0, y_0, tgrid)
            except:
                continue
            x_interpol = interpolate(tabulated_function=x)
            y_interpol = interpolate(tabulated_function=y)
            C1 = C1_score(x_interpol, y_interpol, rho, nnodes, tgrid, T)
            C2 = C2_score(x_interpol, S, T)
            loss = Loss(C1, C2)
            scores.append(loss)
        scores = np.array(scores)
        if len(scores) == 0:
            log.append("Unfortunatelly, no beta satisfies the requirement: y in [0, 1]")
            return False, log
        log.append("Best solution:")
        max_score = np.max(scores)
        log.append("Ф(best_beta) = " + str(max_score))
        best_f = f[np.argmax(scores)]
        best_beta = best_f.beta
        log.append("Best beta = " + str(best_beta))
        log.append("")
        # log.append("Searching solutions for different x0, y0...")
        # log.append("")
        x0_grid = np.array([x_0] + np.linspace(0, 9, 10).tolist())
        y0_grid = np.array([y_0] + np.linspace(0, 0.9, 10).tolist())
        solutions = []
        for x_0, y_0 in zip(x0_grid, y0_grid):
            # log.append("x0 = " + str(x_0))
            # log.append("y0 = " + str(y_0))
            try:
                x, y = diff_equastions(func1, best_f, x_0, y_0, tgrid)
            except:
                # log.append("Y not in [0,1]. Try another x0, y0. Continue...")
                # log.append("")
                continue
            x_interpol = interpolate(tabulated_function=x)
            y_interpol = interpolate(tabulated_function=y)
            C1 = C1_score(x_interpol, y_interpol, rho, nnodes, tgrid, T)
            C2 = C2_score(x_interpol, S, T)
            loss = Loss(C1, C2)
            log.append("x0 = " + str(x_0))
            log.append("y0 = " + str(y_0))
            log.append("C1(x0, y0) = " + str(C1))
            log.append("C2(x0, y0) = " + str(C2))
            log.append("Ф(x0, y0) = " + str(loss))
            log.append("")
            solutions.append(Solution(y, x, S_tab, x_0, y_0, best_beta, C1, C2, loss))
            build_res_graphs(solutions)
            write_to_file(solutions)
        return True, log[:-1]
    else:
        func1 = lambda t, y: z_derivative_interpol(t) * integral_interpol(y)
        try:
            x, y = diff_equastions(func1, f, x_0, y_0, tgrid)
        except:
            log.append("Y not in [0,1]. Try another beta.")
            return False, log
        x_interpol = interpolate(tabulated_function=x)
        y_interpol = interpolate(tabulated_function=y)
        C1 = C1_score(x_interpol, y_interpol, rho, nnodes, tgrid, T)
        C2 = C2_score(x_interpol, S, T)
        loss = Loss(C1, C2)
        log.append("C1_score = " + str(C1))
        log.append("C2_score = " + str(C2))
        log.append("Ф(beta) = " + str(loss))
        solution = Solution(y, x, S_tab, x_0, y_0, f.beta, C1, C2, loss)
        build_res_graphs([solution,])
        write_to_file([solution,])
        return True, log

def initialize_functions(rho_expr, z_expr, S_expr, f_expr, beta=None, beta_min=None, beta_max=None):
    error_log = []
    try:
        rho = FunctionRho(rho_expr)
        rho(1)
    except:
        error_log.append("p(w) is specified incorrectly")
    try:
        z = FunctionSz(z_expr)
        z(1)
    except:
        error_log.append("z(t) is specified incorrectly")
    try:
        S = FunctionSz(S_expr)
        S(1)
    except:
        error_log.append("S(t) is specified incorrectly")

    if error_log:
        return True, error_log

    try:
        f = FunctionF(f_expr, 1, z, S)
        f(1,1)
    except:
        error_log.append("f(beta, z, x, S) is specified incorrectly")

    if error_log:
        return True, error_log

    if beta is not None:
        f = FunctionF(f_expr, beta, z, S)
        return False, (rho, z, S, f)
    else:
        bgrid = np.linspace(beta_min, beta_max, 20)
        fs = [FunctionF(f_expr, beta, z, S) for beta in bgrid]
        return False, (rho, z, S, fs)