# TinyLeastSquares benchmarks

A [BenchmarkDotNet](https://benchmarkdotnet.org/) harness for the solver. It is **deliberately
not part of `TinyLeastSquares.sln`** so the release CI (which builds the solution) never pulls
BenchmarkDotNet or packs this project.

## Run

Timings + allocation (BenchmarkDotNet, ShortRun by default — a couple of minutes):

```
dotnet run -c Release --project benchmarks/Benchmarks -- --filter '*'
```

Deterministic diagnostics — outer-iteration count, residual/Jacobian evaluation count, and
allocation per solve measured via `GC.GetAllocatedBytesForCurrentThread()` (no timing noise,
great for verifying an optimization changed allocation without changing convergence):

```
dotnet run -c Release --project benchmarks/Benchmarks -- diag
```

## Workloads

| Benchmark | Shape | Path exercised |
|-----------|-------|----------------|
| `ExpDecayFit` | dense, n=3, m=256 | Cholesky, several damping rejections |
| `GaussianMixtureFit` | dense, n=18, m=500 | expensive Jacobian, strongly nonlinear |
| `IkReach` | dense, n=6, m=2 | tiny per-iter cost, allocation-dominated (per-frame IK) |
| `SparseChainDirect` | sparse, n=600 | `SparseSolveDirect` (JᵀJ build + sparse Cholesky) |

Each `[Benchmark]` runs a full solve from a fixed start, so the measured cost includes residual +
Jacobian evaluation, the normal-equations build, factorization, and the accept/reject damping loop.
