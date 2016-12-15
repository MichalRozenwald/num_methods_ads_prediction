from django import forms
from .models import Poly

class PolyForm(forms.ModelForm):

    class Meta:
        model = Poly
        fields = ('name', 'coefs_comma_split',)


class rho_z_s_Form(forms.Form):
    rho_coeffs_comma_split = forms.CharField(label='rho_coeffs_comma_split', max_length=1000)
    z_coeffs_comma_split = forms.CharField(label='z_coeffs_comma_split', max_length=1000)
    S_coeffs_comma_split = forms.CharField(label='S_coeffs_comma_split', max_length=1000)

    save_dir_path = forms.CharField(label='save_file_path', max_length=1000)


class file_path_Form(forms.Form):
    file_path = forms.CharField(label='file_path', max_length=1000)


class Cauchy_Form(forms.Form):
    x_0 = forms.CharField(label='x_0', max_length=1000)
    y_0 = forms.CharField(label='y_0', max_length=1000)
    betta = forms.CharField(label='betta', max_length=1000)
    T = forms.CharField(label='T', max_length=1000)

    U_coeffs_comma_split = forms.CharField(label='U_coeffs_comma_split', max_length=1000)
    S_coeffs_comma_split = forms.CharField(label='S_coeffs_comma_split', max_length=1000)
    z_coeffs_comma_split = forms.CharField(label='z_coeffs_comma_split', max_length=1000)
    # f = forms.CharField(label='f', max_length=1000)

    save_dir_path = forms.CharField(label='save_dir_path', max_length=1000)


class Cauchy_auto_Form(forms.Form):
    x_0 = forms.CharField(label='x_0', max_length=1000)
    y_0 = forms.CharField(label='y_0', max_length=1000)
    betta_interval = forms.CharField(label='betta_interval: start, end', max_length=1000)

    T = forms.CharField(label='T', max_length=1000)

    U_coeffs_comma_split = forms.CharField(label='U_coeffs_comma_split', max_length=1000)
    S_coeffs_comma_split = forms.CharField(label='S_coeffs_comma_split', max_length=1000)
    z_coeffs_comma_split = forms.CharField(label='z_coeffs_comma_split', max_length=1000)
    # f = forms.CharField(label='f', max_length=1000)

    save_dir_path = forms.CharField(label='save_dir_path', max_length=1000)




class Ode_manual_Form(forms.Form):
    # x_0 = forms.CharField(label='x_0', max_length=1000)
    # y_0 = forms.CharField(label='y_0', max_length=1000)
    # betta_interval = forms.CharField(label='betta_interval: start, end', max_length=1000)
    # T = forms.CharField(label='T', max_length=1000)
    # U_coeffs_comma_split = forms.CharField(label='U_coeffs_comma_split', max_length=1000)
    # S_coeffs_comma_split = forms.CharField(label='S_coeffs_comma_split', max_length=1000)
    # z_coeffs_comma_split = forms.CharField(label='z_coeffs_comma_split', max_length=1000)
    # f = forms.CharField(label='f', max_length=1000)
    # save_dir_path = forms.CharField(label='save_dir_path', max_length=1000)

    rho = forms.CharField(initial="6*w*(1-w)")
    S = forms.CharField(initial="4*t+cos(t)")
    z = forms.CharField(initial="3*t+sin(t)")
    beta = forms.DecimalField(initial=0.01) # , required=False
    # beta_min = forms.DecimalField(required=False, initial=-1)
    # beta_max = forms.DecimalField(required=False, initial=1)
    f = forms.CharField(initial="beta*(S-x)")
    x_0 = forms.DecimalField(label='x_0', initial=0, required=False)
    y_0 = forms.DecimalField(label='y_0', min_value=0, max_value=1, initial=0)
    T = forms.DecimalField(initial=1)


class Ode_auto_Form(forms.Form):
    rho = forms.CharField(initial="6*w*(1-w)")
    S = forms.CharField(initial="4*t+cos(t)")
    z = forms.CharField(initial="3*t+sin(t)")
    # beta = forms.DecimalField(initial=0.01, required=False)
    beta_min = forms.DecimalField(initial=-1) # required=False,
    beta_max = forms.DecimalField(initial=1)  # required=False,
    f = forms.CharField(initial="beta*(S-x)")
    x_0 = forms.DecimalField(label='x_0', initial=0, required=False)
    y_0 = forms.DecimalField(label='y_0', min_value=0, max_value=1, initial=0)
    T = forms.DecimalField(initial=1)

