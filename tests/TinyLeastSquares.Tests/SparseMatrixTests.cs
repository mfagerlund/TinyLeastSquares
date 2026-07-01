using NUnit.Framework;
using FluentAssertions;
using TinyLeastSquares;

namespace TinyLeastSquares.Tests;

[TestFixture]
public class SparseMatrixTests : TestFixtureBase
{
    [Test]
    public void CreateFromTriplets()
    {
        var triplets = new List<(int row, int col, double value)>
        {
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (2, 0, 4.0),
            (2, 2, 5.0)
        };

        var sparse = SparseMatrix.FromTriplets(3, 3, triplets);

        sparse.Rows.Should().Be(3);
        sparse.Cols.Should().Be(3);
        sparse.NonZeroCount.Should().Be(5);

        sparse.Get(0, 0).Should().Be(1.0);
        sparse.Get(0, 1).Should().Be(0.0); // Not present
        sparse.Get(0, 2).Should().Be(2.0);
        sparse.Get(1, 1).Should().Be(3.0);
        sparse.Get(2, 0).Should().Be(4.0);
        sparse.Get(2, 2).Should().Be(5.0);

        WriteLine($"Sparsity: {sparse.Sparsity:P1}");
    }

    [Test]
    public void CreateWithBuilder()
    {
        var builder = SparseMatrix.CreateBuilder(3, 3);

        builder.BeginRow(0);
        builder.Add(0, 1.0);
        builder.Add(2, 2.0);
        builder.EndRow();

        builder.BeginRow(1);
        builder.Add(1, 3.0);
        builder.EndRow();

        builder.BeginRow(2);
        builder.Add(0, 4.0);
        builder.Add(2, 5.0);
        builder.EndRow();

        var sparse = builder.Build();

        sparse.Get(0, 0).Should().Be(1.0);
        sparse.Get(0, 2).Should().Be(2.0);
        sparse.Get(1, 1).Should().Be(3.0);
        sparse.Get(2, 0).Should().Be(4.0);
        sparse.Get(2, 2).Should().Be(5.0);
    }

    [Test]
    public void MatrixVectorMultiply()
    {
        // A = [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 1), (0, 2, 2),
            (1, 1, 3),
            (2, 0, 4), (2, 2, 5)
        };
        var A = SparseMatrix.FromTriplets(3, 3, triplets);

        var x = new double[] { 1, 2, 3 };
        var y = A.Multiply(x);

        // y = [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
        y[0].Should().BeApproximately(7, 1e-10);
        y[1].Should().BeApproximately(6, 1e-10);
        y[2].Should().BeApproximately(19, 1e-10);
    }

    [Test]
    public void TransposeMultiply()
    {
        // A = [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 1), (0, 2, 2),
            (1, 1, 3),
            (2, 0, 4), (2, 2, 5)
        };
        var A = SparseMatrix.FromTriplets(3, 3, triplets);

        var x = new double[] { 1, 2, 3 };
        var y = A.TransposeMultiply(x);

        // A^T = [[1, 0, 4], [0, 3, 0], [2, 0, 5]]
        // y = [1*1 + 4*3, 3*2, 2*1 + 5*3] = [13, 6, 17]
        y[0].Should().BeApproximately(13, 1e-10);
        y[1].Should().BeApproximately(6, 1e-10);
        y[2].Should().BeApproximately(17, 1e-10);
    }

    [Test]
    public void ComputeJtJ()
    {
        // J = [[1, 2], [3, 4], [5, 6]]  (3x2 matrix)
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 1), (0, 1, 2),
            (1, 0, 3), (1, 1, 4),
            (2, 0, 5), (2, 1, 6)
        };
        var J = SparseMatrix.FromTriplets(3, 2, triplets);

        var JtJ = J.ComputeJtJ();

        // J^T J = [[35, 44], [44, 56]]
        JtJ.Rows.Should().Be(2);
        JtJ.Cols.Should().Be(2);
        JtJ.Get(0, 0).Should().BeApproximately(35, 1e-10);
        JtJ.Get(0, 1).Should().BeApproximately(44, 1e-10);
        JtJ.Get(1, 0).Should().BeApproximately(44, 1e-10);
        JtJ.Get(1, 1).Should().BeApproximately(56, 1e-10);
    }

    [Test]
    public void ComputeJtr()
    {
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 1), (0, 1, 2),
            (1, 0, 3), (1, 1, 4),
            (2, 0, 5), (2, 1, 6)
        };
        var J = SparseMatrix.FromTriplets(3, 2, triplets);

        var r = new double[] { 1, 2, 3 };
        var Jtr = J.ComputeJtr(r);

        // J^T r = [22, 28]
        Jtr[0].Should().BeApproximately(22, 1e-10);
        Jtr[1].Should().BeApproximately(28, 1e-10);
    }

    [Test]
    public void AddDiagonal()
    {
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 1), (0, 1, 2),
            (1, 0, 3), (1, 1, 4)
        };
        var A = SparseMatrix.FromTriplets(2, 2, triplets);

        var B = A.AddDiagonal(10);

        B.Get(0, 0).Should().BeApproximately(11, 1e-10);
        B.Get(0, 1).Should().BeApproximately(2, 1e-10);
        B.Get(1, 0).Should().BeApproximately(3, 1e-10);
        B.Get(1, 1).Should().BeApproximately(14, 1e-10);
    }

    [Test]
    public void ConjugateGradientSolves2x2System()
    {
        // Solve [[4, 1], [1, 3]] x = [1, 2]
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 4), (0, 1, 1),
            (1, 0, 1), (1, 1, 3)
        };
        var A = SparseMatrix.FromTriplets(2, 2, triplets);
        var b = new double[] { 1, 2 };

        var x = SparseLinearSolver.ConjugateGradient(A, b);

        WriteLine($"Solution: x = [{x[0]:F6}, {x[1]:F6}]");

        // Verify Ax = b
        var Ax = A.Multiply(x);
        Ax[0].Should().BeApproximately(b[0], 1e-8);
        Ax[1].Should().BeApproximately(b[1], 1e-8);
    }

    [Test]
    public void PreconditionedCGSolvesLargerSystem()
    {
        // Create a 10x10 tridiagonal SPD matrix
        int n = 10;
        var triplets = new List<(int, int, double)>();

        for (int i = 0; i < n; i++)
        {
            triplets.Add((i, i, 4.0)); // Diagonal
            if (i > 0)
            {
                triplets.Add((i, i - 1, -1.0));
                triplets.Add((i - 1, i, -1.0));
            }
        }

        var A = SparseMatrix.FromTriplets(n, n, triplets);
        var b = new double[n];
        for (int i = 0; i < n; i++) b[i] = 1.0;

        var x = SparseLinearSolver.PreconditionedConjugateGradient(A, b);

        WriteLine($"Solution: [{string.Join(", ", x.Select(v => v.ToString("F4")))}]");

        // Verify Ax ≈ b
        var Ax = A.Multiply(x);
        for (int i = 0; i < n; i++)
        {
            Ax[i].Should().BeApproximately(b[i], 1e-6);
        }
    }

    [Test]
    public void SparseCircleFit()
    {
        // Same circle fit as dense test but using sparse Jacobian
        double trueA = 2.0, trueB = 3.0, trueR = 5.0;
        var points = new List<(double x, double y)>();
        for (int i = 0; i < 8; i++)
        {
            double angle = i * Math.PI / 4;
            points.Add((trueA + trueR * Math.Cos(angle), trueB + trueR * Math.Sin(angle)));
        }

        var parameters = new double[] { 0.0, 0.0, 1.0 };

        SparseResidualEvaluation Evaluate(double[] p)
        {
            double a = p[0], b = p[1], r = p[2];
            var residuals = new double[points.Count];
            var triplets = new List<(int, int, double)>();

            for (int i = 0; i < points.Count; i++)
            {
                double dx = points[i].x - a;
                double dy = points[i].y - b;
                double dist = Math.Sqrt(dx * dx + dy * dy);

                residuals[i] = dist - r;

                if (dist > 1e-10)
                {
                    triplets.Add((i, 0, -dx / dist)); // dri/da
                    triplets.Add((i, 1, -dy / dist)); // dri/db
                }
                triplets.Add((i, 2, -1)); // dri/dr
            }

            var jacobian = SparseMatrix.FromTriplets(points.Count, 3, triplets);
            return new SparseResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.SparseSolve(parameters, Evaluate);

        WriteLine($"Converged in {result.Iterations} iterations, cost={result.FinalCost:E3}");
        WriteLine($"Solution: a = {parameters[0]:F6}, b = {parameters[1]:F6}, r = {parameters[2]:F6}");

        result.Success.Should().BeTrue();
        parameters[0].Should().BeApproximately(trueA, 1e-4);
        parameters[1].Should().BeApproximately(trueB, 1e-4);
        parameters[2].Should().BeApproximately(trueR, 1e-4);
    }

    [Test]
    public void SparseRosenbrockProblem()
    {
        var parameters = new double[] { -1.0, 1.0 };

        SparseResidualEvaluation Evaluate(double[] p)
        {
            double x = p[0], y = p[1];
            var residuals = new double[]
            {
                10 * (y - x * x),
                1 - x
            };
            var triplets = new List<(int, int, double)>
            {
                (0, 0, -20 * x), (0, 1, 10),
                (1, 0, -1)
                // Note: (1, 1) is zero, so we don't add it - sparse!
            };
            var jacobian = SparseMatrix.FromTriplets(2, 2, triplets);
            return new SparseResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.SparseSolve(parameters, Evaluate, new LeastSquaresOptions
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
    /// Simulates a mass-spring chain with very sparse Jacobian.
    /// Each spring residual only touches 2 particles.
    /// </summary>
    [Test]
    public void LargeSparseMassSpringSystem()
    {
        int numParticles = 50;
        int numSprings = numParticles - 1; // Chain
        double restLength = 1.0;

        // Initial positions
        var parameters = new double[numParticles];
        for (int i = 0; i < numParticles; i++)
        {
            parameters[i] = i * 0.5; // Start halfway compressed
        }

        // Target: evenly spaced at restLength apart
        // Residual for spring i: (x[i+1] - x[i]) - restLength

        SparseResidualEvaluation Evaluate(double[] p)
        {
            var residuals = new double[numSprings];
            var triplets = new List<(int, int, double)>();

            for (int i = 0; i < numSprings; i++)
            {
                double dx = p[i + 1] - p[i];
                residuals[i] = dx - restLength;

                // dr_i/dx_i = -1
                // dr_i/dx_{i+1} = 1
                triplets.Add((i, i, -1.0));
                triplets.Add((i, i + 1, 1.0));
            }

            var jacobian = SparseMatrix.FromTriplets(numSprings, numParticles, triplets);
            return new SparseResidualEvaluation(residuals, jacobian);
        }

        var result = NonlinearLeastSquaresSolver.SparseSolve(parameters, Evaluate, new LeastSquaresOptions
        {
            Verbose = true,
            LogCallback = WriteLine
        });

        WriteLine($"Final positions: [{string.Join(", ", parameters.Take(5).Select(v => v.ToString("F2")))} ... {string.Join(", ", parameters.Skip(numParticles - 5).Select(v => v.ToString("F2")))}]");

        result.Success.Should().BeTrue();

        // Check that springs have correct length
        for (int i = 0; i < numSprings; i++)
        {
            double dx = parameters[i + 1] - parameters[i];
            dx.Should().BeApproximately(restLength, 0.01, $"Spring {i} should have length {restLength}");
        }
    }

    [Test]
    public void ToDenseAndBack()
    {
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 1), (0, 2, 2),
            (1, 1, 3),
            (2, 0, 4), (2, 2, 5)
        };
        var sparse = SparseMatrix.FromTriplets(3, 3, triplets);

        var dense = sparse.ToDense();

        dense[0, 0].Should().Be(1);
        dense[0, 1].Should().Be(0);
        dense[0, 2].Should().Be(2);
        dense[1, 0].Should().Be(0);
        dense[1, 1].Should().Be(3);
        dense[1, 2].Should().Be(0);
        dense[2, 0].Should().Be(4);
        dense[2, 1].Should().Be(0);
        dense[2, 2].Should().Be(5);
    }

    [Test]
    public void SparseCholeskyDirectSolvesSPDSystem()
    {
        // Symmetric positive definite: full pattern stored (as ComputeJtJ produces)
        var triplets = new List<(int, int, double)>
        {
            (0, 0, 4), (0, 1, 1),
            (1, 0, 1), (1, 1, 3)
        };
        var A = SparseMatrix.FromTriplets(2, 2, triplets);
        var b = new double[] { 1, 2 };

        var x = SparseLinearSolver.SparseCholeskyDirect(A, b);

        // Verify Ax = b
        var Ax = A.Multiply(x);
        Ax[0].Should().BeApproximately(b[0], 1e-10);
        Ax[1].Should().BeApproximately(b[1], 1e-10);
    }
}
