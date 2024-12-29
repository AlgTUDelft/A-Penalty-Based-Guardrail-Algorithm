from tf_functions import *
from algorithms import *


def initialization(problem_spec, grad_spec, initial_vectors, path):
    N_init = len(initial_vectors)
    for i in range(N_init):
        pm_lb_dict, pm_ub_dict, ipdd_dict, gdpa_dict, pga_dict = {}, {}, {}, {}, {}
        initial_vector = np.array(initial_vectors[i])
        print("Initial vector ", initial_vector)
        suffix = '_'.join(map(str, initial_vector))
        J_pm_lb, constraint_values_pm_lb, var_pm_lb, runtime_pm_lb = standard_penalty_alg(
            num_var=problem_spec["num_var"],
            num_con=problem_spec["num_con"],
            c=problem_spec["c"],
            q=problem_spec["q"],
            ub=problem_spec["ub"],
            lb=problem_spec["lb"], C=grad_spec["C"],
            initial_vector=initial_vector,
            delta=grad_spec["delta"],
            patience=grad_spec["patience"],
            grad_iter_max=grad_spec[
                "grad_iter_max"])
        save(dict_=pm_lb_dict, J=J_pm_lb, f=constraint_values_pm_lb, runtime=runtime_pm_lb, path=path,
             name="pm_lb_init_" + suffix, vars=var_pm_lb)
        J_pm_ub, constraint_values_pm_ub, var_pm_ub, runtime_pm_ub = standard_penalty_alg(
            num_var=problem_spec["num_var"],
            num_con=problem_spec["num_con"],
            c=problem_spec["c"],
            q=problem_spec["q"],
            ub=problem_spec["ub"],
            lb=problem_spec["lb"], C=grad_spec["C_large"],
            initial_vector=initial_vector,
            delta=grad_spec["delta"],
            patience=grad_spec["patience"],
            grad_iter_max=grad_spec[
                "grad_iter_max"])
        save(dict_=pm_ub_dict, J=J_pm_ub, f=constraint_values_pm_ub, runtime=runtime_pm_ub, path=path,
             name="pm_ub_init_" + suffix, vars=var_pm_ub)
        J_ipdd, constraint_values_ipdd, var_ipdd, runtime_ipdd = ipdd(
            num_var=problem_spec["num_var"],
            num_con=problem_spec["num_con"],
            c=problem_spec["c"],
            q=problem_spec["q"], ub=problem_spec["ub"],
            lb=problem_spec["lb"], T=problem_spec["T"],
            rho=1,
            initial_vector=initial_vector,
            initial_lambdas=grad_spec["initial_lambdas"],
            grad_iter_max=grad_spec["grad_iter_max"])
        save(dict_=ipdd_dict, J=J_ipdd, f=constraint_values_ipdd, runtime=runtime_ipdd, path=path,
             name="ipdd_init_" + suffix, vars=var_ipdd)
        J_gdpa, constraint_values_gdpa, var_gdpa, runtime_gdpa = gdpa(num_var=problem_spec["num_var"],
                                                                      num_con=problem_spec["num_con"],
                                                                      c=problem_spec["c"], q=problem_spec["q"],
                                                                      ub=problem_spec["ub"],
                                                                      lb=problem_spec["lb"],
                                                                      T=problem_spec["T"],
                                                                      initial_vector=initial_vector,
                                                                      initial_lambdas=grad_spec["initial_lambdas"],
                                                                      step_size_init=grad_spec["step_size"],
                                                                      perturbation_term=grad_spec["perturbation_term"],
                                                                      beta_init=grad_spec["beta"],
                                                                      gamma=grad_spec["gamma"]
                                                                      )
        save(dict_=gdpa_dict, J=J_gdpa, f=constraint_values_gdpa, runtime=runtime_gdpa, path=path,
             name="gdpa_init_" + suffix, vars=var_gdpa)
        J_pga, constraint_values_pga, var_pga, runtime_pga = pga(num_var=problem_spec["num_var"],
                                                                 num_con=problem_spec["num_con"],
                                                                 c=problem_spec["c"],
                                                                 q=problem_spec["q"], ub=problem_spec["ub"],
                                                                 lb=problem_spec["lb"], T=problem_spec["T"],
                                                                 C=grad_spec["C"],
                                                                 initial_vector=initial_vector,
                                                                 delta=grad_spec["delta"],
                                                                 patience=grad_spec["patience"],
                                                                 grad_iter_max=grad_spec["grad_iter_max"])
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
        pm_lb_dict, pga_dict = {}, {}
        J_pm_lb, constraint_values_pm_lb, vars_lb, runtime_pm_lb = standard_penalty_alg(num_var=problem_spec["num_var"],
                                                                                        num_con=problem_spec["num_con"],
                                                                                        c=problem_spec["c"],
                                                                                        q=problem_spec["q"],
                                                                                        ub=problem_spec["ub"],
                                                                                        lb=problem_spec["lb"], C=C,
                                                                                        initial_vector=grad_spec[
                                                                                            "initial_vector"],
                                                                                        delta=grad_spec["delta"],
                                                                                        patience=grad_spec["patience"],
                                                                                        grad_iter_max=grad_spec[
                                                                                            "grad_iter_max"])
        save(dict_=pm_lb_dict, J=J_pm_lb, f=constraint_values_pm_lb, runtime=runtime_pm_lb, path=path,
             name="pm_lb_" + str(C), vars=vars_lb)
        J_pga, constraint_values_pga, vars_pga, runtime_pga = pga(num_var=problem_spec["num_var"],
                                                                  num_con=problem_spec["num_con"],
                                                                  c=problem_spec["c"],
                                                                  q=problem_spec["q"], ub=problem_spec["ub"],
                                                                  lb=problem_spec["lb"], T=problem_spec["T"], C=C,
                                                                  initial_vector=grad_spec[
                                                                      "initial_vector"],
                                                                  delta=grad_spec["delta"],
                                                                  patience=grad_spec["patience"],
                                                                  grad_iter_max=grad_spec["grad_iter_max"])
        save(dict_=pga_dict, J=J_pga, f=constraint_values_pga, runtime=runtime_pga, path=path,
             name="pga_" + str(C), vars=vars_pga)


def determine_gradient_descent_iterations(problem_spec, grad_spec, grad_iters, path):
    """
    This function determines minimal number of gradient descent iterations leading to convergence.
    :param problem_spec: dict
    :param grad_spec: dict
    :param grad_iters: list
    :param path: Path
    :return: None
    """
    pm_dict = {}
    for grad_iter_max in grad_iters:
        J, constraint_values, vars, runtime = standard_penalty_alg(num_var=problem_spec["num_var"],
                                                                   num_con=problem_spec["num_con"],
                                                                   c=problem_spec["c"],
                                                                   q=problem_spec["q"],
                                                                   ub=problem_spec["ub"],
                                                                   lb=problem_spec["lb"],
                                                                   C=grad_spec["C"],
                                                                   initial_vector=grad_spec[
                                                                       "initial_vector"],
                                                                   delta=grad_spec["delta"],
                                                                   patience=grad_spec["patience"],
                                                                   grad_iter_max=grad_iter_max)
        save(dict_=pm_dict, J=J, f=constraint_values, runtime=runtime, path=path,
             name=f"pm_C={grad_spec['C']}_grad_iter={grad_iter_max}", vars=vars)


def evaluate_gdpa(problem_spec, grad_spec, beta_inits, gammas, path):
    for beta_init, gamma in zip(beta_inits, gammas):
        print("Beta ", beta_init)
        print("Gamma ", gamma)
        gdpa_dict = {}
        J_gdpa, constraint_values_gdpa, var_gdpa, runtime_gdpa = gdpa(
            num_var=problem_spec["num_var"],
            num_con=problem_spec["num_con"],
            c=problem_spec["c"], q=problem_spec["q"],
            ub=problem_spec["ub"],
            lb=problem_spec["lb"],
            T=problem_spec["T"],
            initial_vector=grad_spec["initial_vector"],
            initial_lambdas=grad_spec["initial_lambdas"],
            step_size_init=beta_init * gamma,
            perturbation_term=beta_init * gamma,
            beta_init=beta_init, gamma=gamma
        )
        save(dict_=gdpa_dict, J=J_gdpa, f=constraint_values_gdpa, runtime=runtime_gdpa, path=path,
             name="gdpa_beta_{}".format(beta_init), vars=var_gdpa)
