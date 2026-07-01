using NUnit.Framework;
using FluentAssertions;
using TinyLeastSquares;

namespace TinyLeastSquares.Tests;

[TestFixture]
public class NonlinearLeastSquaresTests : TestFixtureBase
{
    /// <summary>
    /// Test 1: Simple quadratic - minimize (x - 3)^2
    /// Single residual: r(x) = x - 3
    /// Jacobian: J = [1]
    /// Solution: x = 3
    /// </summary>
    [Test]
    public void SolvesSingleVariableQuadratic()
    {
        var parameters = new double[] { 0.0 }; // Start at x = 0

        ResidualEvaluation Evaluate(double[] p)
        {
            double x = p[0];
            // r = x - 3
            var residuals = new double[] { x - 3 };
            // dr/dx = 1
            var jacobian = new double[1, 1];
            jacobian[0, 0] = 1.0;
            return new ResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Evaluate);

        WriteLine($"Converged in {result.Iterations} iterations, cost={result.FinalCost:E3}");
        WriteLine($"Solution: x = {parameters[0]:F6} (expected 3.0)");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(3.0, 1e-6);
        result.FinalCost.Should().BeLessThan(1e-12);
    }

    /// <summary>
    /// Test 2: Two-variable linear system
    /// Minimize ||Ax - b||^2 where A = [[2, 1], [1, 3]], b = [4, 5]
    /// Residuals: r1 = 2x + y - 4, r2 = x + 3y - 5
    /// Solution: x = 1.4, y = 1.2
    /// </summary>
    [Test]
    public void SolvesLinearSystem()
    {
        var parameters = new double[] { 0.0, 0.0 };

        ResidualEvaluation Evaluate(double[] p)
        {
            double x = p[0], y = p[1];
            var residuals = new double[]
            {
                2 * x + y - 4,  // r1
                x + 3 * y - 5   // r2
            };
            // Jacobian is constant for linear problems
            var jacobian = new double[2, 2];
            jacobian[0, 0] = 2; jacobian[0, 1] = 1;  // dr1/dx, dr1/dy
            jacobian[1, 0] = 1; jacobian[1, 1] = 3;  // dr2/dx, dr2/dy
            return new ResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Evaluate);

        WriteLine($"Converged in {result.Iterations} iterations, cost={result.FinalCost:E3}");
        WriteLine($"Solution: x = {parameters[0]:F6}, y = {parameters[1]:F6}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(1.4, 1e-3);
        parameters[1].Should().BeApproximately(1.2, 1e-3);
    }

    /// <summary>
    /// Test 3: Rosenbrock-style problem with two residuals
    /// r1 = 10 * (y - x^2)
    /// r2 = 1 - x
    /// dr1/dx = -20x, dr1/dy = 10
    /// dr2/dx = -1,   dr2/dy = 0
    /// Minimum at (1, 1)
    /// </summary>
    [Test]
    public void SolvesRosenbrockProblem()
    {
        var parameters = new double[] { -1.0, 1.0 }; // Start away from solution

        ResidualEvaluation Evaluate(double[] p)
        {
            double x = p[0], y = p[1];
            var residuals = new double[]
            {
                10 * (y - x * x),  // r1
                1 - x              // r2
            };
            var jacobian = new double[2, 2];
            jacobian[0, 0] = -20 * x; jacobian[0, 1] = 10;  // dr1/dx, dr1/dy
            jacobian[1, 0] = -1;      jacobian[1, 1] = 0;   // dr2/dx, dr2/dy
            return new ResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Evaluate, new LeastSquaresOptions
        {
            MaxIterations = 200
        });

        WriteLine($"Converged in {result.Iterations} iterations, cost={result.FinalCost:E3}");
        WriteLine($"Solution: x = {parameters[0]:F6}, y = {parameters[1]:F6}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(1.0, 1e-4);
        parameters[1].Should().BeApproximately(1.0, 1e-4);
    }

    /// <summary>
    /// Test 4: Circle fitting
    /// Fit (xi - a)^2 + (yi - b)^2 - r^2 = 0 to points on circle centered at (2, 3) with radius 5
    /// Parameters: [a, b, r]
    /// Residual for each point i: ri = sqrt((xi-a)^2 + (yi-b)^2) - r
    /// </summary>
    [Test]
    public void FitsCircleToPoints()
    {
        // Generate points on circle centered at (2, 3) with radius 5
        double trueA = 2.0, trueB = 3.0, trueR = 5.0;
        var points = new List<(double x, double y)>();
        for (int i = 0; i < 8; i++)
        {
            double angle = i * Math.PI / 4;
            points.Add((trueA + trueR * Math.Cos(angle), trueB + trueR * Math.Sin(angle)));
        }

        var parameters = new double[] { 0.0, 0.0, 1.0 }; // Initial guess: center at origin, radius 1

        ResidualEvaluation Evaluate(double[] p)
        {
            double a = p[0], b = p[1], r = p[2];
            var residuals = new double[points.Count];
            var jacobian = new double[points.Count, 3];

            for (int i = 0; i < points.Count; i++)
            {
                double dx = points[i].x - a;
                double dy = points[i].y - b;
                double dist = Math.Sqrt(dx * dx + dy * dy);

                residuals[i] = dist - r;

                // dri/da = -dx/dist, dri/db = -dy/dist, dri/dr = -1
                if (dist > 1e-10)
                {
                    jacobian[i, 0] = -dx / dist;
                    jacobian[i, 1] = -dy / dist;
                }
                jacobian[i, 2] = -1;
            }

            return new ResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Evaluate);

        WriteLine($"Converged in {result.Iterations} iterations, cost={result.FinalCost:E3}");
        WriteLine($"Solution: a = {parameters[0]:F6}, b = {parameters[1]:F6}, r = {parameters[2]:F6}");
        WriteLine($"Expected: a = {trueA}, b = {trueB}, r = {trueR}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(trueA, 1e-5);
        parameters[1].Should().BeApproximately(trueB, 1e-5);
        parameters[2].Should().BeApproximately(trueR, 1e-5);
    }

    /// <summary>
    /// Test 5: Overdetermined system (more equations than unknowns)
    /// Fit y = mx + c to points [(0,1), (1,2), (2,4), (3,5)]
    /// Least squares solution: m = 1.4, c = 0.9
    /// </summary>
    [Test]
    public void FitsLineToPoints()
    {
        var points = new (double x, double y)[] { (0, 1), (1, 2), (2, 4), (3, 5) };
        var parameters = new double[] { 0.0, 0.0 }; // [m, c]

        ResidualEvaluation Evaluate(double[] p)
        {
            double m = p[0], c = p[1];
            var residuals = new double[points.Length];
            var jacobian = new double[points.Length, 2];

            for (int i = 0; i < points.Length; i++)
            {
                double x = points[i].x, y = points[i].y;
                residuals[i] = m * x + c - y;  // predicted - actual
                jacobian[i, 0] = x;  // dr/dm
                jacobian[i, 1] = 1;  // dr/dc
            }

            return new ResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Evaluate);

        WriteLine($"Converged in {result.Iterations} iterations, cost={result.FinalCost:E3}");
        WriteLine($"Solution: m = {parameters[0]:F6}, c = {parameters[1]:F6}");

        result.Success.Should().BeTrue();
        // Least squares solution: m = 1.4, c = 0.9
        parameters[0].Should().BeApproximately(1.4, 1e-5);
        parameters[1].Should().BeApproximately(0.9, 1e-5);
    }

    /// <summary>
    /// Test 6: Exponential fit - y = a * exp(b * x)
    /// Fit to points from y = 2 * exp(0.5 * x)
    /// Residual: ri = a * exp(b * xi) - yi
    /// dri/da = exp(b * xi)
    /// dri/db = a * xi * exp(b * xi)
    /// </summary>
    [Test]
    public void FitsExponentialCurve()
    {
        double trueA = 2.0, trueB = 0.5;
        var points = new List<(double x, double y)>();
        for (int i = 0; i < 6; i++)
        {
            double x = i * 0.5;
            points.Add((x, trueA * Math.Exp(trueB * x)));
        }

        var parameters = new double[] { 1.0, 0.1 }; // Initial guess

        ResidualEvaluation Evaluate(double[] p)
        {
            double a = p[0], b = p[1];
            var residuals = new double[points.Count];
            var jacobian = new double[points.Count, 2];

            for (int i = 0; i < points.Count; i++)
            {
                double x = points[i].x, y = points[i].y;
                double expBx = Math.Exp(b * x);

                residuals[i] = a * expBx - y;
                jacobian[i, 0] = expBx;        // dr/da
                jacobian[i, 1] = a * x * expBx; // dr/db
            }

            return new ResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Evaluate);

        WriteLine($"Converged in {result.Iterations} iterations, cost={result.FinalCost:E3}");
        WriteLine($"Solution: a = {parameters[0]:F6}, b = {parameters[1]:F6}");
        WriteLine($"Expected: a = {trueA}, b = {trueB}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(trueA, 1e-4);
        parameters[1].Should().BeApproximately(trueB, 1e-4);
    }

    /// <summary>
    /// Test 7: Test QR solver option
    /// </summary>
    [Test]
    public void SolvesWithQRDecomposition()
    {
        var parameters = new double[] { 0.0 };

        ResidualEvaluation Evaluate(double[] p)
        {
            double x = p[0];
            var residuals = new double[] { x - 7 };
            var jacobian = new double[1, 1];
            jacobian[0, 0] = 1.0;
            return new ResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.Solve(parameters, Evaluate, new LeastSquaresOptions
        {
            UseQR = true
        });

        WriteLine($"Converged in {result.Iterations} iterations using QR");
        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(7.0, 1e-6);
    }

    /// <summary>
    /// Test 8: Linear solver utilities - Cholesky
    /// </summary>
    [Test]
    public void CholeskyDecompositionWorks()
    {
        // 2x2 positive definite matrix
        var A = new double[2, 2];
        A[0, 0] = 4; A[0, 1] = 2;
        A[1, 0] = 2; A[1, 1] = 5;

        var L = LinearSolver.CholeskyDecomposition(A);

        // Verify L * L^T = A
        var reconstructed = new double[2, 2];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    reconstructed[i, j] += L[i, k] * L[j, k];
                }
            }
        }

        reconstructed[0, 0].Should().BeApproximately(A[0, 0], 1e-10);
        reconstructed[0, 1].Should().BeApproximately(A[0, 1], 1e-10);
        reconstructed[1, 0].Should().BeApproximately(A[1, 0], 1e-10);
        reconstructed[1, 1].Should().BeApproximately(A[1, 1], 1e-10);
    }

    /// <summary>
    /// Test 9: JtJ and Jtr computation
    /// </summary>
    [Test]
    public void JtJAndJtrComputationWorks()
    {
        var J = new double[3, 2];
        J[0, 0] = 1; J[0, 1] = 2;
        J[1, 0] = 3; J[1, 1] = 4;
        J[2, 0] = 5; J[2, 1] = 6;

        var r = new double[] { 1, 2, 3 };

        var JtJ = LinearSolver.ComputeJtJ(J);
        var Jtr = LinearSolver.ComputeJtr(J, r);

        // J^T J = [[35, 44], [44, 56]]
        JtJ[0, 0].Should().BeApproximately(35, 1e-10);
        JtJ[0, 1].Should().BeApproximately(44, 1e-10);
        JtJ[1, 0].Should().BeApproximately(44, 1e-10);
        JtJ[1, 1].Should().BeApproximately(56, 1e-10);

        // J^T r = [22, 28]
        Jtr[0].Should().BeApproximately(22, 1e-10);
        Jtr[1].Should().BeApproximately(28, 1e-10);
    }
}
