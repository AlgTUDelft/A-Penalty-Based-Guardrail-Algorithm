import wandb
import time
from pathlib import Path
from algorithms_nonlinear import mps, standard_penalty_alg, ipdd, gdpa, pga
from helpers import save
from optimization_specs_nonlinear import *
from eval_nonlinear import parameter_C, initialization, evaluate_gdpa

if __name__ == "__main__":
    function = "fun_7"
    path: Path = Path("data/nonlinear").joinpath(function)
    problem_spec = PROBLEM_SPECS[function]
    grad_spec = GRADIENT_SPECS[function]
    #eval_spec = EVAL_SPECS[function]
    mps_dict, pm_lb_dict, pm_ub_dict, ipdd_dict, gdpa_dict, pga_dict = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    J_mps, constraint_values_mps, var_mps, runtime_mps = mps(problem_specs=problem_spec)
    save(dict_=mps_dict, J=J_mps, f=constraint_values_mps, runtime=runtime_mps, path=path, name="mps", vars=var_mps)
    """
    J_pm_lb, constraint_values_pm_lb, var_pm_lb, runtime_pm_lb = standard_penalty_alg(
        problem_specs=problem_spec, grad_specs=grad_spec, C=grad_spec["C"]
    )
    save(dict_=pm_lb_dict, J=J_pm_lb, f=constraint_values_pm_lb, runtime=runtime_pm_lb, path=path,
         name="pm_lb", vars=var_pm_lb)
    J_pm_ub, constraint_values_pm_ub, var_pm_ub, runtime_pm_ub = standard_penalty_alg(
        problem_specs=problem_spec, grad_specs=grad_spec, C=1000
    )
    save(dict_=pm_ub_dict, J=J_pm_ub, f=constraint_values_pm_ub, runtime=runtime_pm_ub, path=path,
         name="pm_ub", vars=var_pm_ub)

    J_ipdd, constraint_values_ipdd, var_ipdd, runtime_ipdd = ipdd(
        problem_specs=problem_spec, grad_specs=grad_spec
    )
    save(dict_=ipdd_dict, J=J_ipdd, f=constraint_values_ipdd, runtime=runtime_ipdd, path=path,
         name="ipdd", vars=var_ipdd)
    """
    J_gdpa, constraint_values_gdpa, var_gdpa, runtime_gdpa = gdpa(
        problem_specs=problem_spec,
        grad_specs=grad_spec,
        step_size_init=0.2475,
        perturbation_term=0.2475,
        beta_init=0.25,
        gamma=0.99,
    )
    save(dict_=gdpa_dict, J=J_gdpa, f=constraint_values_gdpa, runtime=runtime_gdpa, path=path,
         name="gdpa", vars=var_gdpa)
    J_pga, constraint_values_pga, var_pga, runtime_pga = pga(
        problem_specs=problem_spec, grad_specs=grad_spec, C=grad_spec["C"]
    )
    save(dict_=pga_dict, J=J_pga, f=constraint_values_pga, runtime=runtime_pga, path=path,
         name="pga", vars=var_pga)
    """
    parameter_C(problem_spec=problem_spec, grad_spec=grad_spec, Cs=eval_spec["Cs"],
                path=path.joinpath("parameter_C"))
    initialization(problem_spec=problem_spec, grad_spec=grad_spec, initial_vectors=eval_spec["initial_vectors"],
                   path=path.joinpath("initialization"))
    evaluate_gdpa(problem_spec=problem_spec, grad_spec=grad_spec, beta_inits=[0.9, 0.75, 0.5, 0.25, 0.1],
                  gammas=[0.99, 0.99, 0.99, 0.99, 0.99], path=path.joinpath("evaluate_gdpa"))
    """
