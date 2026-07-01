# TinyLeastSquares

Nano-sized nonlinear least squares for .NET.

A small, dependency-light **Levenberg-Marquardt** solver for `min Σ rᵢ(x)²` — the
.NET cousin of Ceres / `scipy.optimize.least_squares`. Dense **and** sparse,
analytic **or** finite-difference Jacobians, plus an L-BFGS minimizer for scalar
energies. No autodiff; you bring the residuals (and, if you have them, the
derivatives).

- **One dependency:** [CSparse](https://www.nuget.org/packages/CSparse) (Tim Davis's sparse Cholesky port), MIT.
- **net8.0**, nullable-enabled, SIMD-accelerated inner loops.
- MIT licensed.

## Install

```
dotnet add package TinyLeastSquares
```

## Quick start

```csharp
using TinyLeastSquares;

// minimize sum of squared residuals for:  x + y = 3,  2x - y = 1
ResidualEvaluation Evaluate(double[] p)
{
    double x = p[0], y = p[1];

    var residuals = new[]
    {
        x + y - 3,   // want == 0
        2*x - y - 1  // want == 0
    };

    // Jacobian dr_i/dp_j (analytic)
    var jacobian = new double[2, 2];
    jacobian[0, 0] = 1;  jacobian[0, 1] =  1;
    jacobian[1, 0] = 2;  jacobian[1, 1] = -1;

    return new ResidualEvaluation(residuals, jacobian);
}

var p = new double[] { 0, 0 };
var result = NonlinearLeastSquaresSolver.Solve(p, Evaluate);

Console.WriteLine($"x={p[0]}, y={p[1]} ({result.ConvergenceReason})");
```

## No Jacobian? Use finite differences

If you can't (or don't want to) derive derivatives, pass a residuals-only
function and the solver approximates the Jacobian numerically (forward or
central differences, SciPy-style step sizes):

```csharp
double[] Residuals(double[] p)
{
    double m = p[0], c = p[1];
    var r = new double[points.Length];
    for (int i = 0; i < points.Length; i++)
        r[i] = m * points[i].x + c - points[i].y;   // line fit
    return r;
}

var p = new double[] { 0, 0 };
NonlinearLeastSquaresSolver.Solve(p, Residuals);                              // forward (default)
NonlinearLeastSquaresSolver.Solve(p, Residuals, null, FiniteDifferenceScheme.Central);
```

Analytic Jacobians are faster and more accurate — prefer them when available.

## Sparse problems

For large problems where each residual touches only a few parameters (physics
constraints, bundle-adjustment-like structures), build a sparse Jacobian and use
`SparseSolve` (iterative, Jacobi-PCG) or `SparseSolveDirect` (direct sparse
Cholesky via CSparse — typically much faster on fixed-sparsity systems):

```csharp
SparseResidualEvaluation Evaluate(double[] p)
{
    var residuals = new double[numSprings];
    var triplets = new List<(int, int, double)>();
    for (int i = 0; i < numSprings; i++)
    {
        residuals[i] = (p[b[i]] - p[a[i]]) - restLength;
        triplets.Add((i, a[i], -1.0));   // only 2 non-zeros per row
        triplets.Add((i, b[i],  1.0));
    }
    var J = SparseMatrix.FromTriplets(numSprings, numParticles, triplets);
    return new SparseResidualEvaluation(residuals, J);
}

NonlinearLeastSquaresSolver.SparseSolveDirect(positions, Evaluate);
```

| Scenario | Use |
|----------|-----|
| < ~100 parameters, dense coupling | `Solve` |
| Large, mostly-zero Jacobian | `SparseSolve` |
| Large, fixed sparsity, want speed | `SparseSolveDirect` |

## Options

```csharp
new LeastSquaresOptions
{
    MaxIterations     = 100,
    CostTolerance     = 1e-6,
    ParamTolerance    = 1e-6,
    GradientTolerance = 1e-6,
    InitialDamping    = 1e-3,   // Levenberg-Marquardt λ
    AdaptiveDamping   = true,   // λ /= 10 on accept, *= 10 on reject
    UseQR             = false,  // QR instead of Cholesky for the normal equations
    TrustRegionRadius = double.PositiveInfinity,
    Verbose           = false,
    LogCallback       = Console.WriteLine,
}
```

## What's in the box

| Type | Purpose |
|------|---------|
| `NonlinearLeastSquaresSolver` | Levenberg-Marquardt — `Solve`, `SparseSolve`, `SparseSolveDirect`, finite-difference overloads |
| `SparseMatrix` | CSR sparse matrix — triplets/builder, `Multiply`, `ComputeJtJ`, `ComputeJtr`, `AddDiagonal` |
| `SparseLinearSolver` | CG, Jacobi-PCG, IC-CG, direct sparse Cholesky |
| `LinearSolver` | Dense Cholesky / QR |
| `FiniteDifference` | Numeric Jacobian (forward / central) |
| `LBFGSSolver` | Limited-memory BFGS for scalar `f(x)` with gradient (not least-squares) |

## Algorithm

Levenberg-Marquardt interpolates between Gauss-Newton (λ→0, fast near the
solution) and gradient descent (λ→∞, robust far away). Each iteration solves the
damped normal equations

```
(JᵀJ + λI) δ = −Jᵀr
```

with λ adapted automatically on step accept/reject.

## Scope

TinyLeastSquares does **unconstrained** nonlinear least squares. Bounds (box)
constraints and a zero-allocation reusable workspace are on the [roadmap](ROADMAP.md);
general nonlinear (equality/inequality) constraints are intentionally out of scope.

## License

MIT — see [LICENSE](LICENSE).
