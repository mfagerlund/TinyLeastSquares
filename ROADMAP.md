# Roadmap

## v1.0 (current)
- Levenberg-Marquardt nonlinear least squares (dense + sparse).
- Sparse linear solvers: CG, Jacobi-PCG, IC-CG, direct sparse Cholesky (CSparse).
- Dense linear algebra: Cholesky, QR.
- **Finite-difference Jacobians** (forward / central) via residuals-only overloads.
- L-BFGS unconstrained minimizer.

## Performance

A BenchmarkDotNet harness lives in `benchmarks/Benchmarks` (kept out of the solution so
release CI never builds it). Run `dotnet run -c Release --project benchmarks/Benchmarks -- --filter '*'`
for timings, or `... -- diag` for deterministic iteration/eval counts and per-solve allocation.

Done:
- **JᵀJ / Jᵀr are built once per outer iteration and reused across the inner damping loop**
  (only λ changes on the diagonal). JᵀJ is built lazily — after the convergence checks — so a
  converging iteration never pays for it. Wins scale with step rejections: ~-15% allocation and
  ~-25% time on a rejection-heavy dense fit; flat (no regression) on fast-converging problems.

Next perf levers (surfaced by the benchmark):
- **Residuals-only trial evaluation.** During damping/line-search trials the solver only needs the
  residual vector, but the `ResidualFunction` contract forces the caller to build the full Jacobian
  every call. On expensive-Jacobian problems (e.g. the Gaussian-mixture benchmark) those discarded
  Jacobians dominate the cost. A residuals-only trial path would help most non-trivial fits.
- **Sparse symbolic factorization caching.** `SparseCholeskyDirect` redoes AMD ordering + symbolic
  factorization every iteration though the pattern is fixed; `SparseMatrix.ComputeJtJ` rebuilds and
  sorts a triplet list every iteration. Caching the pattern / symbolic step is a large sparse win.

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
