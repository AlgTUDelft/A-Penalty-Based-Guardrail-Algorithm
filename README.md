# A Penalty-based Guardrail Algorithm
This repository presents the Penalty-Based Guardrail Algorithm (PGA) for solving large-scale, nonlinear, and nonconvex constrained minimization problems with increasing objective function and non-decreasing inequality constraints under tight time constraints. 

## Key Idea
PGA adapts the traditional penalty-based algorithm -- which applies a low-strength penalty term for constraint violations –- by introducing a guardrail variable that dynamically adjusts the right-hand side of the constraints. This variable is updated iteratively based
on historical constraint values. This approach helps balance constraint violations and the minimization of the objective function. 

## Comparison Benchmarks
We compare PGA against four well-known five algorithms:
- Standard penalty-based algorithm with a small value of the penalty parameter ($PA_{C \searrow}$),
- Standard penalty-based algorithm with a large penalty parameter ($PA_{C \nearrow}$),
- Increasing penalty dual decomposiion (IPDD) algorithm,
- Gradiet descent perturbed ascent (GDPA) algorithm,
- Baseline mathematical programming solver (MPS).

## Evaluation Domains
- Three linear evaluation domains,
- Three nonlinear evaluation domains,
- Neural network-based control of a district heating system domain.

## Installation and Operation
To install this package:
```
git clone https://github.com/KStepanovic1/A-Penalty-Based-Guardrail-Algorithm.git
cd /A-Penalty-Based-Guardrail-Algorithm
pip install -r requirements.txt
```
To run algorithms on linear domains:
```
python -m src.linear.algorithms_run.run
```
To run algorithms on nonlinear domains:
```
python -m src.nonlinear.algorithms_run.run
```
To run algorithm on the neural network-based control of a district heating system domain: 
```
python -m src.dnn.algorithms_run.run
```
