import numpy as np
from algorithms_nonlinear import standard_penalty_alg, ipdd, gdpa, pga
from helpers import save

def evaluate_gdpa(problem_spec, grad_spec, beta_inits, gammas, path):
    for beta_init, gamma in zip(beta_inits, gammas):
        print("Beta ", beta_init)
        print("Gamma ", gamma)
        gdpa_dict = {}
        J_gdpa, constraint_values_gdpa, var_gdpa, runtime_gdpa = gdpa(
            problem_specs=problem_spec,
            grad_specs=grad_spec,
            step_size_init=beta_init*gamma,
            perturbation_term=beta_init*gamma,
            beta_init=beta_init,
            gamma=gamma,
        )
        save(dict_=gdpa_dict, J=J_gdpa, f=constraint_values_gdpa, runtime=runtime_gdpa, path=path,
             name="gdpa_beta_{}".format(beta_init), vars=var_gdpa)

def initialization(problem_spec, grad_spec, initial_vectors, path):
    N_init = len(initial_vectors)
    for i in range(N_init):
        pm_lb_dict, pm_ub_dict, ipdd_dict, gdpa_dict, pga_dict = {}, {}, {}, {}, {}
        initial_vector = np.array(initial_vectors[i])
        print("Initial vector ", initial_vector)
        suffix = '_'.join(map(str, initial_vector))
        J_pm_lb, constraint_values_pm_lb, var_pm_lb, runtime_pm_lb = standard_penalty_alg(
            problem_specs=problem_spec, grad_specs=grad_spec, C=grad_spec["C"]
        )
        save(dict_=pm_lb_dict, J=J_pm_lb, f=constraint_values_pm_lb, runtime=runtime_pm_lb, path=path,
             name="pm_lb_init_" + suffix, vars=var_pm_lb)
        J_pm_ub, constraint_values_pm_ub, var_pm_ub, runtime_pm_ub = standard_penalty_alg(
            problem_specs=problem_spec, grad_specs=grad_spec, C=1000
        )
        save(dict_=pm_ub_dict, J=J_pm_ub, f=constraint_values_pm_ub, runtime=runtime_pm_ub, path=path,
             name="pm_ub_init_" + suffix, vars=var_pm_ub)
        J_ipdd, constraint_values_ipdd, var_ipdd, runtime_ipdd = ipdd(
            problem_specs=problem_spec, grad_specs=grad_spec
        )
        save(dict_=ipdd_dict, J=J_ipdd, f=constraint_values_ipdd, runtime=runtime_ipdd, path=path,
             name="ipdd_init_" + suffix, vars=var_ipdd)
        J_gdpa, constraint_values_gdpa, var_gdpa, runtime_gdpa = gdpa(
            problem_specs=problem_spec,
            grad_specs=grad_spec,
            step_size_init=0.891,
            perturbation_term=0.891,
            beta_init=0.9,
            gamma=0.99,
        )
        save(dict_=gdpa_dict, J=J_gdpa, f=constraint_values_gdpa, runtime=runtime_gdpa, path=path,
             name="gdpa_init_" + suffix, vars=var_gdpa)
        J_pga, constraint_values_pga, var_pga, runtime_pga = pga(
            problem_specs=problem_spec, grad_specs=grad_spec, C=grad_spec["C"]
        )
        save(dict_=pga_dict, J=J_pga, f=constraint_values_pga, runtime=runtime_pga, path=path,
             name="pga_init_" + suffix, vars=var_pga)


def parameter_C(problem_spec, grad_spec, Cs, path):
    """
    This function answers two questions:
    - How do different strengths of penalty parameter C influence the penalty algorithm with small value of penalty parameter?
    - How these different strengths influence the PGA performance?
    :param problem_spec: dict
    :param grad_spec: dict
    :param C: list
    :param path: Path
    :return: None
    """
    for C in Cs:
        print("C={}".format(C))
        pm_dict, pga_dict = {}, {}
        J_pm, constraint_values_pm, var_pm, runtime_pm = standard_penalty_alg(problem_specs=problem_spec,
                                                                              grad_specs=grad_spec, C=C)
        save(dict_=pm_dict, J=J_pm, f=constraint_values_pm, runtime=runtime_pm, path=path,
             name="pm_" + str(C), vars=var_pm)
        J_pga, constraint_values_pga, var_pga, runtime_pga = pga(problem_specs=problem_spec, grad_specs=grad_spec, C=C)
        save(dict_=pga_dict, J=J_pga, f=constraint_values_pga, runtime=runtime_pga, path=path,
             name="pga_" + str(C), vars=var_pga)
