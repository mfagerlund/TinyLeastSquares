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

## Gallery

Every animation below is a **real solve** — TinyLeastSquares runs the Levenberg-Marquardt solver at
generation time and the motion is replayed as a self-contained, looping SVG. Open on GitHub to see
them move; regenerate with `dotnet run --project samples/Demos`.

<p align="center">
  <img src="https://raw.githubusercontent.com/mfagerlund/TinyLeastSquares/master/gallery/ik-target.svg" width="440" alt="A 4-link IK arm following a moving target"><br>
  <em>Inverse kinematics — a 4-link arm's tip tracks a moving target; LM is re-solved and
  warm-started every frame (analytic Jacobian).</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mfagerlund/TinyLeastSquares/master/gallery/curve-fit.svg" width="300" alt="Exponential curve fitting">
  <img src="https://raw.githubusercontent.com/mfagerlund/TinyLeastSquares/master/gallery/circle-fit.svg" width="300" alt="Circle fitting">
</p>
<p align="center">
  <em>Curve fit <code>y = a·e^(−b·x) + c</code> via a finite-difference Jacobian &middot;
  geometric circle fit (cx, cy, r).</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mfagerlund/TinyLeastSquares/master/gallery/registration.svg" width="360" alt="Rigid registration aligning a shape onto a target"><br>
  <em>Rigid registration — solve rotation + translation to align a shape onto a noisy target.</em>
</p>

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

Analytic Jacobians are faster and more accurate — prefer them when available. And you don't have to
derive one by hand: describe your residual to an LLM and let
[gradient-script](https://www.npmjs.com/package/gradient-script) (`npm i gradient-script`) do the
symbolic differentiation for you — it generates ready-to-paste **C#** derivative code (also
TypeScript / JavaScript / Python) from a small DSL, which you drop straight into your
`ResidualEvaluation`.

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
