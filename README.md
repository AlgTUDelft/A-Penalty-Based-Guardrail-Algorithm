# A Penalty-based Guardrail Algorithm
This repository presents the Penalty-Based Guardrail Algorithm (PGA) for solving large-scale, nonlinear, and nonconvex constrained minimization problems with increasing objective function and non-decreasing inequality constraints under tight time constraints. 

## Key Idea
PGA adapts the traditional penalty-based algorithm -- which applies a low-strength penalty term for constraint violations –- by introducing a guardrail variable that dynamically adjusts the right-hand side of the constraints. This variable is updated iteratively based
on historical constraint values. This approach helps balance constraint violations and the minimization of the objective function. 

## Comparison Benchmarks
We compare PGA against four well-known first-order algorithms:
- Standard penalty-based algorithm with a small value of the penalty parameter ($PA_{C \searrow}$),
- Standard penalty-based algorithm with a large penalty parameter ($PA_{C \nearrow}$),
- Increasing penalty dual decomposiion (IPDD) algorithm,
- Gradiet descent perturbed ascent (GDPA) algorithm,
- Baseline mathematical programming solver (MPS).

## Evaluation Domains
- Three linear evaluation domains,
- Three nonlinear evaluation domains,
- Real-world control of a district heating system domain. 
