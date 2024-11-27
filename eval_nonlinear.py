from algorithms_nonlinear import standard_penalty_alg, pga
from helpers import save


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
