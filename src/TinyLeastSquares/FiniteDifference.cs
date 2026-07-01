using System;

namespace TinyLeastSquares;

/// <summary>
/// Finite-difference scheme used to approximate a Jacobian numerically.
/// </summary>
public enum FiniteDifferenceScheme
{
    /// <summary>Forward differences (n extra evaluations per Jacobian, first-order accurate).</summary>
    Forward,

    /// <summary>Central differences (2n extra evaluations per Jacobian, second-order accurate).</summary>
    Central
}

/// <summary>
/// Delegate for computing just the residual vector given current parameters,
/// with no analytic Jacobian. Pair with the finite-difference overloads of
/// <see cref="NonlinearLeastSquaresSolver"/> when derivatives are unavailable.
/// </summary>
public delegate double[] ResidualsOnlyFunction(double[] parameters);

/// <summary>
/// Approximates a dense Jacobian of a residual function using finite differences.
/// Use when analytic derivatives are unavailable or inconvenient.
///
/// Step sizes follow the SciPy <c>least_squares</c> convention: a relative step of
/// sqrt(eps) for forward differences and cbrt(eps) for central differences,
/// scaled by max(1, |x_j|).
/// </summary>
public static class FiniteDifference
{
    /// <summary>Machine epsilon for double precision.</summary>
    private const double Eps = 2.220446049250313e-16;

    private static readonly double ForwardRelStep = Math.Sqrt(Eps);   // ~1.49e-8
    private static readonly double CentralRelStep = Math.Cbrt(Eps);   // ~6.06e-6

    /// <summary>
    /// Wraps a residuals-only function into a <see cref="ResidualFunction"/> whose Jacobian
    /// is computed by finite differences.
    /// </summary>
    /// <param name="residualsFn">Function returning the residual vector for given parameters.</param>
    /// <param name="scheme">Forward or central differences.</param>
    /// <param name="relStep">Relative step size; pass 0 to use the scheme default.</param>
    public static ResidualFunction MakeJacobian(
        ResidualsOnlyFunction residualsFn,
        FiniteDifferenceScheme scheme = FiniteDifferenceScheme.Forward,
        double relStep = 0)
    {
        return p => new ResidualEvaluation(
            residualsFn(p),
            ComputeJacobian(residualsFn, p, scheme, relStep));
    }

    /// <summary>
    /// Computes the dense finite-difference Jacobian (m residuals x n parameters) of
    /// <paramref name="residualsFn"/> at the point <paramref name="parameters"/>.
    /// The point is left unchanged on return.
    /// </summary>
    public static double[,] ComputeJacobian(
        ResidualsOnlyFunction residualsFn,
        double[] parameters,
        FiniteDifferenceScheme scheme = FiniteDifferenceScheme.Forward,
        double relStep = 0)
    {
        int n = parameters.Length;
        var x = (double[])parameters.Clone();
        var r0 = residualsFn(x);
        int m = r0.Length;
        var jac = new double[m, n];

        for (int j = 0; j < n; j++)
        {
            double xj = x[j];
            double h = StepSize(xj, scheme, relStep);

            if (scheme == FiniteDifferenceScheme.Central)
            {
                x[j] = xj + h;
                var rp = residualsFn(x);
                x[j] = xj - h;
                var rm = residualsFn(x);
                x[j] = xj;

                double inv = 1.0 / (2.0 * h);
                for (int i = 0; i < m; i++)
                    jac[i, j] = (rp[i] - rm[i]) * inv;
            }
            else // Forward
            {
                x[j] = xj + h;
                var rp = residualsFn(x);
                x[j] = xj;

                double inv = 1.0 / h;
                for (int i = 0; i < m; i++)
                    jac[i, j] = (rp[i] - r0[i]) * inv;
            }
        }

        return jac;
    }

    private static double StepSize(double x, FiniteDifferenceScheme scheme, double relStep)
    {
        double rs = relStep > 0
            ? relStep
            : (scheme == FiniteDifferenceScheme.Central ? CentralRelStep : ForwardRelStep);
        return rs * Math.Max(1.0, Math.Abs(x));
    }
}

public static partial class NonlinearLeastSquaresSolver
{
    /// <summary>
    /// Solves min sum(r_i(x)^2) when only residuals are available: the Jacobian is
    /// approximated by finite differences. Convenient but less accurate and slower
    /// (n or 2n extra residual evaluations per iteration) than supplying an analytic
    /// <see cref="ResidualFunction"/>.
    /// </summary>
    /// <param name="initialParams">Initial parameter values (modified in place).</param>
    /// <param name="residualsFn">Function returning the residual vector.</param>
    /// <param name="options">Solver options (optional).</param>
    /// <param name="scheme">Finite-difference scheme (forward by default).</param>
    /// <param name="relStep">Relative FD step size; 0 uses the scheme default.</param>
    public static LeastSquaresResult Solve(
        double[] initialParams,
        ResidualsOnlyFunction residualsFn,
        LeastSquaresOptions? options = null,
        FiniteDifferenceScheme scheme = FiniteDifferenceScheme.Forward,
        double relStep = 0)
    {
        return Solve(initialParams, FiniteDifference.MakeJacobian(residualsFn, scheme, relStep), options);
    }
}
