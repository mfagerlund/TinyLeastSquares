using NUnit.Framework;
using FluentAssertions;
using TinyLeastSquares;

namespace TinyLeastSquares.Tests;

[TestFixture]
public class FiniteDifferenceTests : TestFixtureBase
{
    /// <summary>
    /// The finite-difference Jacobian should closely match an analytic one on a
    /// smooth nonlinear function (exponential model).
    /// </summary>
    [Test]
    public void FiniteDifferenceMatchesAnalyticJacobian()
    {
        var xs = new double[] { 0.0, 0.5, 1.0, 1.5 };

        double[] Residuals(double[] p)
        {
            double a = p[0], b = p[1];
            var r = new double[xs.Length];
            for (int i = 0; i < xs.Length; i++)
                r[i] = a * Math.Exp(b * xs[i]) - 1.0; // arbitrary target
            return r;
        }

        var point = new double[] { 2.0, 0.5 };

        var analytic = new double[xs.Length, 2];
        for (int i = 0; i < xs.Length; i++)
        {
            double expBx = Math.Exp(point[1] * xs[i]);
            analytic[i, 0] = expBx;                    // dr/da
            analytic[i, 1] = point[0] * xs[i] * expBx; // dr/db
        }

        var fdCentral = FiniteDifference.ComputeJacobian(Residuals, point, FiniteDifferenceScheme.Central);
        var fdForward = FiniteDifference.ComputeJacobian(Residuals, point, FiniteDifferenceScheme.Forward);

        for (int i = 0; i < xs.Length; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                fdCentral[i, j].Should().BeApproximately(analytic[i, j], 1e-6);
                fdForward[i, j].Should().BeApproximately(analytic[i, j], 1e-4);
            }
        }
    }

    /// <summary>
    /// ComputeJacobian must not mutate the point it is evaluated at.
    /// </summary>
    [Test]
    public void ComputeJacobianDoesNotMutateInput()
    {
        var point = new double[] { 1.5, -2.0, 3.0 };
        var copy = (double[])point.Clone();

        FiniteDifference.ComputeJacobian(p => new[] { p[0] * p[1] + p[2] }, point, FiniteDifferenceScheme.Central);

        point.Should().Equal(copy);
    }

    /// <summary>
    /// Line fit through the residuals-only (forward FD) overload should reach the
    /// same least-squares solution as the analytic path.
    /// </summary>
    [Test]
    public void SolvesLineFitWithForwardDifferences()
    {
        var points = new (double x, double y)[] { (0, 1), (1, 2), (2, 4), (3, 5) };
        var parameters = new double[] { 0.0, 0.0 };

        double[] Residuals(double[] p)
        {
            double m = p[0], c = p[1];
            var r = new double[points.Length];
            for (int i = 0; i < points.Length; i++)
                r[i] = m * points[i].x + c - points[i].y;
            return r;
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Residuals);

        WriteLine($"FD forward: iters={result.Iterations}, m={parameters[0]:F6}, c={parameters[1]:F6}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(1.4, 1e-4);
        parameters[1].Should().BeApproximately(0.9, 1e-4);
    }

    /// <summary>
    /// Exponential fit through the residuals-only (central FD) overload.
    /// </summary>
    [Test]
    public void SolvesExponentialFitWithCentralDifferences()
    {
        double trueA = 2.0, trueB = 0.5;
        var points = new List<(double x, double y)>();
        for (int i = 0; i < 6; i++)
        {
            double x = i * 0.5;
            points.Add((x, trueA * Math.Exp(trueB * x)));
        }

        var parameters = new double[] { 1.0, 0.1 };

        double[] Residuals(double[] p)
        {
            double a = p[0], b = p[1];
            var r = new double[points.Count];
            for (int i = 0; i < points.Count; i++)
                r[i] = a * Math.Exp(b * points[i].x) - points[i].y;
            return r;
        }

        var result = NonlinearLeastSquaresSolver.Solve(
            parameters, Residuals, new LeastSquaresOptions { MaxIterations = 200 },
            FiniteDifferenceScheme.Central);

        WriteLine($"FD central: iters={result.Iterations}, a={parameters[0]:F6}, b={parameters[1]:F6}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(trueA, 1e-3);
        parameters[1].Should().BeApproximately(trueB, 1e-3);
    }

    /// <summary>
    /// Circle fit with finite differences (nonlinear, 3 parameters).
    /// </summary>
    [Test]
    public void SolvesCircleFitWithFiniteDifferences()
    {
        double trueA = 2.0, trueB = 3.0, trueR = 5.0;
        var points = new List<(double x, double y)>();
        for (int i = 0; i < 8; i++)
        {
            double angle = i * Math.PI / 4;
            points.Add((trueA + trueR * Math.Cos(angle), trueB + trueR * Math.Sin(angle)));
        }

        var parameters = new double[] { 0.0, 0.0, 1.0 };

        double[] Residuals(double[] p)
        {
            double a = p[0], b = p[1], r = p[2];
            var res = new double[points.Count];
            for (int i = 0; i < points.Count; i++)
            {
                double dx = points[i].x - a;
                double dy = points[i].y - b;
                res[i] = Math.Sqrt(dx * dx + dy * dy) - r;
            }
            return res;
        }

        var result = NonlinearLeastSquaresSolver.Solve(
            parameters, Residuals, null, FiniteDifferenceScheme.Central);

        WriteLine($"FD circle: iters={result.Iterations}, a={parameters[0]:F5}, b={parameters[1]:F5}, r={parameters[2]:F5}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(trueA, 1e-3);
        parameters[1].Should().BeApproximately(trueB, 1e-3);
        parameters[2].Should().BeApproximately(trueR, 1e-3);
    }
}
