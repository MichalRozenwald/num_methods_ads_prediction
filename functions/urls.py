from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.welcome, name='welcome'),
    url(r'^new_poly$', views.new_poly, name='new_poly'),
    url(r'^rho_z_s_Tabulate$', views.rho_z_s_Tabulate, name='rho_z_s_Tabulate'),
    url(r'^tabulate_integral_from_file_path$', views.tabulate_integral_from_file_path, name='tabulate_integral_from_file_path'),
    url(r'^Cauchy$', views.Cauchy, name='Cauchy'),
    url(r'^Cauchy_auto$', views.Cauchy_auto, name='Cauchy_auto'),
    url(r'^solve_ode_manual$', views.solve_ode_manual, name='solve_ode_manual'),
    url(r'^solve_ode_auto$', views.solve_ode_auto, name='solve_ode_auto'),


    # url(r'^tabulate', views.tabulate, name='tabulate'),
    # url(r'^func_poly/(?P<pk>\d+)/tabulate/$', views.tabulate_poly, name='tabulate_poly'),

]