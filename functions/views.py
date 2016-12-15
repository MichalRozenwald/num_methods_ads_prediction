# -*- coding: utf-8 -*-

import subprocess

from .forms import PolyForm, rho_z_s_Form, file_path_Form, Cauchy_Form, Cauchy_auto_Form, Ode_manual_Form, Ode_auto_Form
from .models import Poly
from django.shortcuts import render, get_object_or_404
# from django.contrib.auth.decorators import login_required
# from django.utils import timezone
# from django.shortcuts import redirect
# from django.http.response import JsonResponse, HttpResponse
# from django.core.exceptions import ObjectDoesNotExist
import numpy as np
import pandas as pd
from pandas.tools.plotting import table

from functions.structures import *
from functions.methods import *


def welcome(request):
    return render(request, 'functions/welcome.html', {})


def new_poly(request):
    if request.method == "POST":
        form = PolyForm(request.POST)
        if form.is_valid():
            poly = form.save(commit=False)
            if (isBADInput(poly.coefs_comma_split)):
                form = PolyForm()
                return render(request, 'functions/func_edit.html', {'form': form, 'error': "BAD INPUT. Try again."})

            poly.coeff = [x.strip() for x in poly.coefs_comma_split.split(',')]
            poly.save()
            print(poly.coeff)
            print(poly)
            # return redirect('functions.views.post_detail', pk=poly.pk)
            # TABULATE
            step_len = 0.016
            x_steps = np.arange(0, 1, step_len)
            tabu_df = Tabulate(poly, x_steps)
            tabu_df_html_path = '/home/michal/Projects/ads_chm/staticfiles/table.html'
            tabu_df.to_html(tabu_df_html_path)
            # tabu_df_table = table(tabu_df)
            context = {
                'func_str': poly.PolyPrint(),
                'func_poly': poly,
                'tabu_df': tabu_df.as_matrix(),
                # 'tabu_df_table': tabu_df_table,
                'tabu_df_html_path': tabu_df_html_path,
            }
            return render(request, 'functions/func_tabu_show.html', context=context)

    else:
        form = PolyForm()
    return render(request, 'functions/func_edit.html', {'form': form, 'error': ''})



def tabulate(request):
    # function need to be passed
    return render(request, 'functions/welcome.html', {})


def tabulate_poly(request, pk):
    poly = get_object_or_404(Poly, pk=pk)
    step_len = 0.016
    x_steps = np.arange(0, 1, step_len)
    tabu_df = Tabulate(poly, x_steps)
    context = {
        'func_str': poly.PolyPrint(),
        'func_poly': poly,
        'tabu_df' : tabu_df,
    }
    return render(request, 'functions/func_tabu_show.html', context=context)



def rho_z_s_Tabulate(request):
    if request.method == "POST":
        form = rho_z_s_Form(request.POST)
        if form.is_valid():
            rho_coeff_str = form.cleaned_data['rho_coeffs_comma_split']
            z_coeff_str =  form.cleaned_data['z_coeffs_comma_split']
            S_coeff_str = form.cleaned_data['S_coeffs_comma_split']

            if (isBADInput(rho_coeff_str) or isBADInput(z_coeff_str) or isBADInput(S_coeff_str)):
                form = rho_z_s_Form()
                return render(request, 'functions/rho_z_s_edit.html', {'form': form, 'error': "BAD INPUT. Try again."})

            rho_coeff = [x.strip() for x in rho_coeff_str.split(',')]
            z_coeff = [x.strip() for x in z_coeff_str.split(',')]
            S_coeff = [x.strip() for x in S_coeff_str.split(',')]

            rho = Polynomial(rho_coeff)
            z = Polynomial(z_coeff)
            S = Polynomial(S_coeff)

            step_len = 0.016
            steps = np.arange(0, 1, step_len)
            rho_tabu_df = Tabulate(rho, steps)
            z_tabu_df = Tabulate(z, steps)
            S_tabu_df = Tabulate(S, steps)


            save_dir_path = str(form.cleaned_data['save_dir_path'])

            try:
                tabu_save_path_rho = save_dir_path + '/Tabu_rho.cvs'
                tabu_save_path_z = save_dir_path + '/Tabu_z.cvs'
                tabu_save_path_S = save_dir_path + '/Tabu_S.cvs'
                save_to_CSV(rho_tabu_df, tabu_save_path_rho)
                save_to_CSV(z_tabu_df, tabu_save_path_z)
                save_to_CSV(S_tabu_df, tabu_save_path_S)

                context = {
                    'rho_str': rho.PolyPrint(),
                    'z_str': z.PolyPrint(),
                    'S_str': S.PolyPrint(),
                    'rho_tabu_df': rho_tabu_df.as_matrix(),
                    'z_tabu_df': z_tabu_df.as_matrix(),
                    'S_tabu_df': S_tabu_df.as_matrix(),
                    'tabu_save_path_rho': tabu_save_path_rho,
                    'tabu_save_path_z': tabu_save_path_z,
                    'tabu_save_path_s': tabu_save_path_S,


                }
                return render(request, 'functions/rho_z_s_tabu_show.html', context=context)
            except FileNotFoundError:
                form = rho_z_s_Form()
                return render(request, 'functions/rho_z_s_edit.html', {'form': form, 'error': "BAD INPUT. Try again. WRONG File Path"})


    else:
        form = rho_z_s_Form()
    return render(request, 'functions/rho_z_s_edit.html', {'form': form, 'error': ''})


# ====================================================================================================================================
# ============================================= Tabulate Integral ====================================================================
# ====================================================================================================================================

# file_path = '/home/michal/Projects/ads_chm/staticfiles/Tabu_rho.cvs'

def tabulate_integral_from_file_path(request):
    if request.method == "POST":
        form = file_path_Form(request.POST)
        if form.is_valid():
            file_path = form.cleaned_data['file_path']
            try:
                tabu_df = open_CSV(file_path)
                tabulate_integral_result = tabulate_integral(tabu_df)
                interpolation_coefficients_results = interpolation_coefficients(tabu_df)

                context = {
                    'file_path': file_path,
                    'tabulate_integral_result': tabulate_integral_result,
                    'interpolation_coefficients_results': interpolation_coefficients_results,
                }
                return render(request, 'functions/tabulate_integral_from_file_path_show.html', context=context)
            except FileNotFoundError:
                form = file_path_Form()
                return render(request, 'functions/tabulate_integral_from_file_path_edit.html', {'form': form, 'error': "BAD INPUT. Try again. WRONG File Path"})


    else:
        form = file_path_Form()
    return render(request, 'functions/tabulate_integral_from_file_path_edit.html', {'form': form, 'error': ""})


def Cauchy(request):
    if request.method == "POST":
        form = Cauchy_Form(request.POST)
        if form.is_valid():
            x_0 = form.cleaned_data['x_0']
            y_0 = form.cleaned_data['y_0']
            betta = form.cleaned_data['betta']
            T = form.cleaned_data['T']

            U_coeff_str = form.cleaned_data['U_coeffs_comma_split']
            S_coeff_str = form.cleaned_data['S_coeffs_comma_split']
            z_coeff_str = form.cleaned_data['z_coeffs_comma_split']

            if (isBADInput(x_0) or isBADInput(y_0) or isBADInput(betta) or isBADInput(T) or
                    isBADInput(U_coeff_str) or isBADInput(S_coeff_str) or isBADInput(z_coeff_str)):
                form = Cauchy_Form()
                return render(request, 'functions/Cauchy_edit.html', {'form': form, 'error': "BAD INPUT. Try again."})

            x_0 = int(form.cleaned_data['x_0'])
            y_0 = int(form.cleaned_data['y_0'])
            betta = float(form.cleaned_data['betta'])
            betta_neg = 0 - betta
            T = int(form.cleaned_data['T'])

            U_coeff = [x.strip() for x in U_coeff_str.split(',')]
            S_coeff = [x.strip() for x in S_coeff_str.split(',')]
            z_coeff = [x.strip() for x in z_coeff_str.split(',')]

            U = Polynomial(U_coeff)
            S = Polynomial(S_coeff)
            z = Polynomial(z_coeff)

            # fun_x, x_0,
            # fun_x = z' * U(t)
            # DEFINE z = ax + b => z' = b = z.coeff[0]
            fun_x = U * int(z.coeff[0])
            # fun_y = (считаем) - позставляю функцию z (линейно) и b  - считаем
            # DEFINE f function == fun_y
            # fun_y = betta * x - betta * z
            fun_y = Polynomial([betta, 0]) + (z * int(betta_neg))  # f -- polinomial
            t_0 = 0
            tau = 0.16

            x_y_tabu_df = euler_method_tow_dim(fun_x, x_0, fun_y, y_0, t_0, T, tau)

            try:
                save_dir_path = str(form.cleaned_data['save_dir_path'])
                Cauchy_cvs_file_path = save_dir_path + '/' + "Cauchy_Euler_tabu.csv"
                save_to_CSV(x_y_tabu_df, Cauchy_cvs_file_path)

                context = {
                    'x_0': x_0,
                    'y_0': y_0,
                    'betta': betta,
                    'T': T,

                    'U_str': U.PolyPrint(),
                    'S_str': S.PolyPrint(),
                    'z_str': z.PolyPrint(),
                    'x_y_tabu_df': x_y_tabu_df.as_matrix(),
                    'Cauchy_cvs_file_path': Cauchy_cvs_file_path,
                }
                return render(request, 'functions/Cauchy_show.html', context=context)
            except FileNotFoundError:
                form = Cauchy_Form()
                return render(request, 'functions/Cauchy_edit.html', {'form': form, 'error': "BAD INPUT. Try again. WRONG File Path"})


    else:
        form = Cauchy_Form()
    return render(request, 'functions/Cauchy_edit.html', {'form': form, 'error': ""})



def Cauchy_auto(request):
    if request.method == "POST":
        form = Cauchy_auto_Form(request.POST)
        if form.is_valid():
            x_0 = form.cleaned_data['x_0']
            y_0 = form.cleaned_data['y_0']
            T = form.cleaned_data['T']

            U_coeff_str = form.cleaned_data['U_coeffs_comma_split']
            S_coeff_str = form.cleaned_data['S_coeffs_comma_split']
            z_coeff_str = form.cleaned_data['z_coeffs_comma_split']
            betta_interval_str = form.cleaned_data['betta_interval']


            if (isBADInput(x_0) or isBADInput(y_0) or isBADInput(betta_interval_str) or isBADInput(T) or
                    isBADInput(U_coeff_str) or isBADInput(S_coeff_str) or isBADInput(z_coeff_str)):
                form = Cauchy_auto_Form()
                return render(request, 'functions/Cauchy_auto_edit.html', {'form': form, 'error': "BAD INPUT. Try again."})

            x_0 = int(form.cleaned_data['x_0'])
            y_0 = int(form.cleaned_data['y_0'])
            # betta = float(form.cleaned_data['betta'])
            # betta_neg = 0 - betta
            T = int(form.cleaned_data['T'])

            U_coeff = [x.strip() for x in U_coeff_str.split(',')]
            S_coeff = [x.strip() for x in S_coeff_str.split(',')]
            z_coeff = [x.strip() for x in z_coeff_str.split(',')]
            betta_interval = [int(x.strip()) for x in betta_interval_str.split(',')]

            if (len(betta_interval) != 2):
                form = Cauchy_auto_Form()
                return render(request, 'functions/Cauchy_auto_edit.html',
                              {'form': form, 'error': "BAD INPUT. Try again. WRONG Betta interval"})

            U = Polynomial(U_coeff)
            S = Polynomial(S_coeff)
            z = Polynomial(z_coeff)


            # fun_x, x_0,
            # fun_x = z' * U(t)
            # DEFINE z = ax + b => z' = b = z.coeff[0]
            fun_x = U * int(z.coeff[0])
            # fun_y = (считаем) - позставляю функцию z (линейно) и b  - считаем
            # DEFINE f function == fun_y
            # fun_y = betta * x - betta * z
            betta_values = np.arange(betta_interval[1], betta_interval[0], 0.16)
            df_list = []
            for betta in betta_interval:  # WORK WITH PARAMS
                betta_neg = 0 - betta
                fun_y = Polynomial([betta, 0]) + (z * int(betta_neg))  # f -- polinomial
                t_0 = 0
                tau = 0.16

                x_y_tabu_df = euler_method_tow_dim(fun_x, x_0, fun_y, y_0, t_0, T, tau)
                df_list.append(x_y_tabu_df)


            try:
                save_dir_path = str(form.cleaned_data['save_dir_path'])
                Cauchy_cvs_file_path = save_dir_path + '/' + "Cauchy_Euler_tabu.csv"
                save_to_CSV(x_y_tabu_df, Cauchy_cvs_file_path)

                context = {
                    'x_0': x_0,
                    'y_0': y_0,
                    'betta_interval': betta_interval,
                    'T': T,

                    'U_str': U.PolyPrint(),
                    'S_str': S.PolyPrint(),
                    'z_str': z.PolyPrint(),
                    'x_y_tabu_df': x_y_tabu_df.as_matrix(),
                    'df_list': df_list,
                    'Cauchy_cvs_file_path': Cauchy_cvs_file_path,
                }
                return render(request, 'functions/Cauchy_auto_show.html', context=context)
            except FileNotFoundError:
                form = Cauchy_auto_Form()
                return render(request, 'functions/Cauchy_auto_edit.html', {'form': form, 'error': "BAD INPUT. Try again. WRONG File Path"})


    else:
        form = Cauchy_auto_Form()
    return render(request, 'functions/Cauchy_auto_edit.html', {'form': form, 'error': ""})




# ====================================================================================================================================
# ============================================= Solver ODE ====================================================================
# ====================================================================================================================================
#
# def main(request):
#     if request.method == 'POST':
#         form = ParamsFormNew(request.POST)
#         mode = request.POST.getlist('mode')
#         err, manual, automat = set_mode(mode)
#         if err:
#             args = {'form' : form, 'errs' : err}
#             return render(request, 'newmain.html', args)
#         if not form.is_valid():
#             args = {'form' : form, 'errs' : [str(form.errors)]}
#             return render(request, 'newmain.html', args)
#         if form.is_valid():
#             rho_expr = form.cleaned_data['rho']
#             S_expr = form.cleaned_data['S']
#             z_expr = form.cleaned_data['z']
#             if manual:
#                 beta = float(form.cleaned_data['beta'])
#                 x_0 = float(form.cleaned_data['x_0'])
#             if automat:
#                 beta_min = float(form.cleaned_data['beta_min'])
#                 beta_max = float(form.cleaned_data['beta_max'])
#             f_expr = form.cleaned_data['f']
#             y_0 = float(form.cleaned_data['y_0'])
#             T = float(form.cleaned_data['T'])
#             if manual:
#                 are_errors, output = initialize_functions(rho_expr, z_expr, S_expr, f_expr, beta=beta)
#             else:
#                 are_errors, output = initialize_functions(rho_expr, z_expr, S_expr, f_expr,
#                                                           beta_min=beta_min, beta_max=beta_max)
#             if are_errors:
#                 for e in output:
#                     err.append(e)
#                 args = {'form' : form, 'automat' : automat, 'manual' : manual,
#                         'errs' : err}
#                 return render(request, 'newmain.html', args)
#             else:
#                 rho, z, S, f = output
#                 err, err_rho = verify_rho(rho, err)
#                 if err:
#                     args = {'form' : form, 'automat' : automat, 'manual' : manual,
#                     'errs' : err, 'err_rho' : err_rho}
#                     return render(request, 'newmain.html', args)
#                 else:
#                     if manual:
#                         is_good_res, log_res = solver(False, rho, z, S, f, y_0, x_0, T)
#                     else:
#                         x_0 = S(0)
#                         is_good_res, log_res = solver(True, rho, z, S, f, y_0, x_0, T)
#                     args = {'form' : form, 'automat' : automat, 'manual' : manual, 'res' : log_res,
#                             'is_good_res' : is_good_res, 'rho' : True, 'z' : True, 'S' : True}
#                     return render(request, 'newmain.html', args)
#         args = {'form' : form, 'automat' : automat, 'manual' : manual, 'err' : err}
#         return render(request, 'newmain.html', args)
#     else:
#         form = ParamsFormNew()
#         context = {'form' : form}
#     return render(request, 'newmain.html', args)
#





def solve_ode_manual(request):
    err = []
    if request.method == 'POST':
        form = Ode_manual_Form(request.POST)
        # mode = request.POST.getlist('mode')
        # err, manual, automat = set_mode(mode)
        # if err:
        #     args = {'form' : form, 'errs' : err}
        #     return render(request, 'newmain.html', args)
        if not form.is_valid():
            context = {
                'form' : form,
                'error' : [str(form.errors)]
            }
            return render(request, 'functions/Ode_manual_edit.html', context)
        else:  # form.is_valid():
            rho_expr = form.cleaned_data['rho']
            S_expr = form.cleaned_data['S']
            z_expr = form.cleaned_data['z']
            x_0 = float(form.cleaned_data['x_0'])
            y_0 = float(form.cleaned_data['y_0'])
            f_expr = form.cleaned_data['f']
            T = float(form.cleaned_data['T'])
            # if manual:
            beta = float(form.cleaned_data['beta'])
            are_errors, output = initialize_functions(rho_expr, z_expr, S_expr, f_expr, beta=beta)
            # if automat:
            #     beta_min = float(form.cleaned_data['beta_min'])
            #     beta_max = float(form.cleaned_data['beta_max'])
            #     are_errors, output = initialize_functions(rho_expr, z_expr, S_expr, f_expr, beta_min = beta_min, beta_max = beta_max)
            if are_errors:
                err = []
                for e in output:
                    err.append(e)
                context = {
                    'form' : form,
                    'error' : err  #, 'automat' : automat, 'manual' : manual,
                }
                return render(request, 'functions/Ode_manual_edit.html', context)
            else:
                rho, z, S, f = output
                err, err_rho = verify_rho(rho, err)
                if err_rho:
                    err.append("Error: wrong input of rho function")
                if err:
                    context = {
                        'form' : form,
                        'error' : err
                    } # 'automat' : automat, 'manual' : manual, 'errs' : err, 'err_rho' : err_rho
                    return render(request, 'functions/Ode_manual_edit.html', context)
                else:
                    #if manual:
                    is_good_res, log_res = solver(False, rho, z, S, f, y_0, x_0, T)
                    # else:
                    #     x_0 = S(0)
                    #     is_good_res, log_res = solver(True, rho, z, S, f, y_0, x_0, T)
                    context = {
                        'form' : form,
                        'res' : log_res,
                        'is_good_res' : is_good_res,
                        'rho' : rho_expr,
                        'z' : z_expr,
                        'S' : S_expr,
                        'f' : f_expr,
                        'x_0' : x_0,
                        'y_0': y_0,
                        'betta' : beta,
                        'T' : T
                    } #'automat' : automat, 'manual' : manual,
                    # return render(request, 'functions/Ode_manual_edit.html', context)
                    return render(request, 'functions/Ode_manual_show.html', context=context)

    else:
        form = Ode_manual_Form()
    return render(request, 'functions/Ode_manual_edit.html', {'form': form, 'error': ""})







def solve_ode_auto(request):
    err = []
    if request.method == 'POST':
        form = Ode_auto_Form(request.POST)
        if not form.is_valid():
            context = {
                'form' : form,
                'error' : [str(form.errors)]
            }
            return render(request, 'functions/Ode_auto_edit.html', context)
        else:  # form.is_valid():
            rho_expr = form.cleaned_data['rho']
            S_expr = form.cleaned_data['S']
            z_expr = form.cleaned_data['z']
            x_0 = float(form.cleaned_data['x_0'])
            y_0 = float(form.cleaned_data['y_0'])
            f_expr = form.cleaned_data['f']
            T = float(form.cleaned_data['T'])
            beta_min = float(form.cleaned_data['beta_min'])
            beta_max = float(form.cleaned_data['beta_max'])
            are_errors, output = initialize_functions(rho_expr, z_expr, S_expr, f_expr, beta_min = beta_min, beta_max = beta_max)
            if are_errors:
                err = []
                for e in output:
                    err.append(e)
                context = {
                    'form' : form,
                    'error' : err
                }
                return render(request, 'functions/Ode_auto_edit.html', context)
            else:
                rho, z, S, f = output
                err, err_rho = verify_rho(rho, err)
                if err_rho:
                    err.append("Error: wrong input of rho function")
                if err:
                    context = {
                        'form' : form,
                        'error' : err
                    }
                    return render(request, 'functions/Ode_auto_edit.html', context)
                else:
                    x_0 = S(0)
                    is_good_res, log_res = solver(True, rho, z, S, f, y_0, x_0, T)
                    context = {
                        'form' : form,
                        'res' : log_res,
                        'is_good_res' : is_good_res,
                        'rho' : rho_expr,
                        'z' : z_expr,
                        'S' : S_expr,
                        'f' : f_expr,
                        'x_0' : x_0,
                        'y_0': y_0,
                        'beta_min': beta_min,
                        'beta_max' : beta_max,
                        'T' : T
                    }
                    return render(request, 'functions/Ode_auto_show.html', context=context)

    else:
        form = Ode_auto_Form()
    return render(request, 'functions/Ode_auto_edit.html', {'form': form, 'error': ""})

