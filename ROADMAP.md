# Roadmap

## v1.0 (current)
- Levenberg-Marquardt nonlinear least squares (dense + sparse).
- Sparse linear solvers: CG, Jacobi-PCG, IC-CG, direct sparse Cholesky (CSparse).
- Dense linear algebra: Cholesky, QR.
- **Finite-difference Jacobians** (forward / central) via residuals-only overloads.
- L-BFGS unconstrained minimizer.

## Deferred / considered

### Bounds (box) constraints
Support `lo <= x <= hi` on parameters, à la SciPy `least_squares(bounds=...)`.
Likely trust-region-reflective or projected steps. Bigger change than v1 features;
tracked for v1.1.

### Zero-allocation reusable workspace
Pre-allocate all working buffers in a reusable solver/context object so repeated
minimizations of the same-shaped problem run allocation-free. This is the idea
behind keir/tinysolver (integrated into Ceres as `ceres::TinySolver`), which
reports large speedups for small dense repeated solves. Perf feature, not a new
capability.

### Sparse finite differences
FD currently produces a dense Jacobian. A sparsity-pattern-aware FD (graph
coloring) would make FD usable for large sparse problems without dense cost.

### General nonlinear constraints
Equality/inequality constraints via interior-point / SQP (cf.
tomstewart89/tinyoptimizer) are explicitly **out of scope** — that is a different
problem class and would belong in a separate package, not here.
