using NUnit.Framework;
using FluentAssertions;
using TinyLeastSquares;

namespace TinyLeastSquares.Tests;

[TestFixture]
public class LBFGSTests : TestFixtureBase
{
    /// <summary>
    /// Minimize a smooth non-quadratic objective f(x,y) = (x-3)^2 + (y+1)^4.
    /// Minimum at (3, -1). Exercises genuine nonlinear (non-quadratic) descent,
    /// which this solver's Armijo-backtracking line search handles reliably.
    /// (Note: the classic Rosenbrock valley from a far start is beyond an
    /// Armijo-only line search — see ROADMAP for a Wolfe line search.)
    /// </summary>
    [Test]
    public void MinimizesSmoothNonQuadratic()
    {
        var parameters = new double[] { 0.0, 2.0 };

        (double, double[]) Evaluate(double[] p)
        {
            double x = p[0], y = p[1];
            double f = Math.Pow(x - 3, 2) + Math.Pow(y + 1, 4);
            var g = new double[2];
            g[0] = 2 * (x - 3);
            g[1] = 4 * Math.Pow(y + 1, 3);
            return (f, g);
        }

        var result = LBFGSSolver.Solve(parameters, Evaluate, new LBFGSOptions
        {
            MaxIterations = 500
        });

        WriteLine($"L-BFGS: iters={result.Iterations}, f={result.FinalCost:E3}, reason={result.ConvergenceReason}");
        WriteLine($"Solution: x={parameters[0]:F6}, y={parameters[1]:F6}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(3.0, 1e-4);
        // Quartic gradient vanishes near the minimum, so y converges more loosely.
        parameters[1].Should().BeApproximately(-1.0, 1e-2);
    }

    /// <summary>
    /// Minimize a simple quadratic bowl f(x) = sum (x_i - i)^2.
    /// </summary>
    [Test]
    public void MinimizesQuadraticBowl()
    {
        int n = 5;
        var parameters = new double[n];

        (double, double[]) Evaluate(double[] p)
        {
            double f = 0;
            var g = new double[n];
            for (int i = 0; i < n; i++)
            {
                double d = p[i] - i;
                f += d * d;
                g[i] = 2 * d;
            }
            return (f, g);
        }

        var result = LBFGSSolver.Solve(parameters, Evaluate);

        WriteLine($"L-BFGS bowl: iters={result.Iterations}, f={result.FinalCost:E3}");

        result.Success.Should().BeTrue();
        for (int i = 0; i < n; i++)
            parameters[i].Should().BeApproximately(i, 1e-4);
    }
}
