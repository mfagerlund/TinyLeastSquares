using System;
using System.Numerics;

namespace TinyLeastSquares;

/// <summary>
/// Configuration options for the nonlinear least squares solver.
/// </summary>
public class LeastSquaresOptions
{
    /// <summary>Maximum number of outer iterations.</summary>
    public int MaxIterations { get; set; } = 100;

    /// <summary>Stop when cost change is below this threshold.</summary>
    public double CostTolerance { get; set; } = 1e-6;

    /// <summary>Stop when parameter change is below this threshold.</summary>
    public double ParamTolerance { get; set; } = 1e-6;

    /// <summary>Stop when gradient norm is below this threshold.</summary>
    public double GradientTolerance { get; set; } = 1e-6;

    /// <summary>Number of line search steps when not using adaptive damping.</summary>
    public int LineSearchSteps { get; set; } = 10;

    /// <summary>Initial Levenberg-Marquardt damping factor.</summary>
    public double InitialDamping { get; set; } = 1e-3;

    /// <summary>Use adaptive damping (Levenberg-Marquardt style).</summary>
    public bool AdaptiveDamping { get; set; } = true;

    /// <summary>Factor by which to increase damping on rejection.</summary>
    public double DampingIncreaseFactor { get; set; } = 10;

    /// <summary>Factor by which to decrease damping on acceptance.</summary>
    public double DampingDecreaseFactor { get; set; } = 10;

    /// <summary>Maximum inner iterations for damping adjustment.</summary>
    public int MaxInnerIterations { get; set; } = 10;

    /// <summary>Use QR decomposition instead of Cholesky.</summary>
    public bool UseQR { get; set; } = false;

    /// <summary>Trust region radius (Infinity = no limit).</summary>
    public double TrustRegionRadius { get; set; } = double.PositiveInfinity;

    /// <summary>Enable verbose logging via callback.</summary>
    public bool Verbose { get; set; } = false;

    /// <summary>Optional callback for logging.</summary>
    public Action<string>? LogCallback { get; set; }
}

/// <summary>
/// Result object returned by the nonlinear least squares solver.
/// </summary>
public class LeastSquaresResult
{
    /// <summary>Whether the solver converged successfully.</summary>
    public bool Success { get; init; }

    /// <summary>Number of iterations performed.</summary>
    public int Iterations { get; init; }

    /// <summary>Final cost (sum of squared residuals).</summary>
    public double FinalCost { get; init; }

    /// <summary>Reason for convergence or termination.</summary>
    public string ConvergenceReason { get; init; } = "";

    /// <summary>Computation time in milliseconds.</summary>
    public double ComputationTimeMs { get; init; }
}

/// <summary>
/// Result of evaluating residuals and Jacobian.
/// </summary>
public readonly struct ResidualEvaluation
{
    /// <summary>Residual values (m elements).</summary>
    public readonly double[] Residuals;

    /// <summary>Jacobian matrix (m residuals x n parameters).</summary>
    public readonly double[,] Jacobian;

    /// <summary>Creates an evaluation from a residual vector and its dense Jacobian.</summary>
    public ResidualEvaluation(double[] residuals, double[,] jacobian)
    {
        Residuals = residuals;
        Jacobian = jacobian;
    }
}

/// <summary>
/// Delegate for computing residuals and Jacobian given current parameters.
/// </summary>
/// <param name="parameters">Current parameter values (modified in place by solver)</param>
/// <returns>Residuals and Jacobian matrix</returns>
public delegate ResidualEvaluation ResidualFunction(double[] parameters);

/// <summary>
/// Result of evaluating residuals and sparse Jacobian.
/// </summary>
public readonly struct SparseResidualEvaluation
{
    /// <summary>Residual values (m elements).</summary>
    public readonly double[] Residuals;

    /// <summary>Sparse Jacobian matrix (m residuals x n parameters).</summary>
    public readonly SparseMatrix Jacobian;

    /// <summary>Creates an evaluation from a residual vector and its sparse Jacobian.</summary>
    public SparseResidualEvaluation(double[] residuals, SparseMatrix jacobian)
    {
        Residuals = residuals;
        Jacobian = jacobian;
    }
}

/// <summary>
/// Delegate for computing residuals and sparse Jacobian given current parameters.
/// </summary>
public delegate SparseResidualEvaluation SparseResidualFunction(double[] parameters);

/// <summary>
/// Nonlinear least squares solver using Levenberg-Marquardt algorithm.
/// This implementation requires explicit Jacobians - no automatic differentiation.
/// (Finite-difference Jacobian overloads live in FiniteDifference.cs.)
/// </summary>
public static partial class NonlinearLeastSquaresSolver
{
    /// <summary>
    /// Solves a nonlinear least squares problem: min sum(r_i(x)^2)
    /// </summary>
    /// <param name="initialParams">Initial parameter values (will be modified in place)</param>
    /// <param name="residualFn">Function computing residuals and Jacobian</param>
    /// <param name="options">Solver options (optional)</param>
    /// <returns>Result containing convergence info</returns>
    public static LeastSquaresResult Solve(
        double[] initialParams,
        ResidualFunction residualFn,
        LeastSquaresOptions? options = null)
    {
        return SolveCore(
            initialParams,
            p => { var e = residualFn(p); return (e.Residuals, e.Jacobian); },
            (jacobian, residuals, lambda) => SolveNormalEquations((double[,])jacobian, residuals, lambda, (options ?? new LeastSquaresOptions()).UseQR),
            (jacobian, residuals) => LinearSolver.ComputeJtr((double[,])jacobian, residuals),
            null,
            options);
    }

    /// <summary>
    /// Solves a nonlinear least squares problem with sparse Jacobian: min sum(r_i(x)^2)
    /// More efficient for large problems where Jacobian is mostly zeros.
    /// </summary>
    /// <param name="initialParams">Initial parameter values (will be modified in place)</param>
    /// <param name="residualFn">Function computing residuals and sparse Jacobian</param>
    /// <param name="options">Solver options (optional)</param>
    /// <returns>Result containing convergence info</returns>
    public static LeastSquaresResult SparseSolve(
        double[] initialParams,
        SparseResidualFunction residualFn,
        LeastSquaresOptions? options = null)
    {
        return SolveCore(
            initialParams,
            p => { var e = residualFn(p); return (e.Residuals, e.Jacobian); },
            (jacobian, residuals, lambda) => SolveSparseNormalEquations((SparseMatrix)jacobian, residuals, lambda),
            (jacobian, residuals) => ((SparseMatrix)jacobian).ComputeJtr(residuals),
            jacobian => { var j = (SparseMatrix)jacobian; return $", nnz={j.NonZeroCount}, sparsity={j.Sparsity:P1}"; },
            options);
    }

    /// <summary>
    /// Same as <see cref="SparseSolve"/> but uses direct sparse Cholesky (CSparse.NET
    /// via <see cref="SparseLinearSolver.SparseCholeskyDirect"/>) for the per-iter
    /// normal-equations solve. Typical 7-10x faster than PCG on well-structured
    /// problems (e.g. large fixed-sparsity systems at ~20k vars).
    /// </summary>
    public static LeastSquaresResult SparseSolveDirect(
        double[] initialParams,
        SparseResidualFunction residualFn,
        LeastSquaresOptions? options = null)
    {
        return SolveCore(
            initialParams,
            p => { var e = residualFn(p); return (e.Residuals, e.Jacobian); },
            (jacobian, residuals, lambda) => SolveSparseNormalEquationsDirect((SparseMatrix)jacobian, residuals, lambda),
            (jacobian, residuals) => ((SparseMatrix)jacobian).ComputeJtr(residuals),
            jacobian => { var j = (SparseMatrix)jacobian; return $", nnz={j.NonZeroCount}, sparsity={j.Sparsity:P1}"; },
            options);
    }

    private static LeastSquaresResult SolveCore(
        double[] initialParams,
        Func<double[], (double[] Residuals, object Jacobian)> evaluate,
        Func<object, double[], double, double[]> solveNormal,
        Func<object, double[], double[]> computeGradient,
        Func<object, string>? extraLogInfo,
        LeastSquaresOptions? options)
    {
        options ??= new LeastSquaresOptions();
        var startTime = DateTime.UtcNow;

        var parameters = initialParams;
        double prevCost = double.PositiveInfinity;
        double lambda = options.InitialDamping;

        void Log(string message)
        {
            if (options.Verbose)
            {
                options.LogCallback?.Invoke(message);
            }
        }

        for (int iter = 0; iter < options.MaxIterations; iter++)
        {
            var (residuals, jacobian) = evaluate(parameters);
            double cost = ComputeCost(residuals);

            var Jtr = computeGradient(jacobian, residuals);
            double gradientNorm = Math.Sqrt(ComputeSumOfSquares(Jtr));

            Log($"Iteration {iter}: cost={FormatNumber(cost)}, ||∇||={FormatNumber(gradientNorm)}" +
                (extraLogInfo != null ? extraLogInfo(jacobian) : "") +
                (options.AdaptiveDamping ? $", λ={FormatNumber(lambda)}" : ""));

            if (gradientNorm < options.GradientTolerance)
            {
                return CreateResult(true, iter, cost, "Gradient tolerance reached", startTime);
            }

            if (Math.Abs(prevCost - cost) < options.CostTolerance)
            {
                return CreateResult(true, iter, cost, "Cost tolerance reached", startTime);
            }

            if (cost < options.CostTolerance)
            {
                return CreateResult(true, iter, cost, "Cost below threshold", startTime);
            }

            double[]? delta;
            bool accepted = false;
            int innerIterations = 0;

            while (!accepted && innerIterations < options.MaxInnerIterations)
            {
                try
                {
                    delta = solveNormal(jacobian, residuals, options.AdaptiveDamping ? lambda : 0);
                }
                catch (Exception e)
                {
                    Log($"  Linear solver failed: {e.Message}");
                    return CreateResult(false, iter, cost, $"Linear solver failed: {e.Message}", startTime);
                }

                double deltaNorm = Math.Sqrt(ComputeSumOfSquares(delta));

                if (deltaNorm > options.TrustRegionRadius)
                {
                    double scale = options.TrustRegionRadius / deltaNorm;
                    for (int i = 0; i < delta.Length; i++)
                    {
                        delta[i] *= scale;
                    }
                    deltaNorm = options.TrustRegionRadius;
                }

                if (deltaNorm < options.ParamTolerance)
                {
                    return CreateResult(true, iter, cost, "Parameter tolerance reached", startTime);
                }

                if (options.AdaptiveDamping)
                {
                    var originalParams = (double[])parameters.Clone();
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        parameters[i] += delta[i];
                    }

                    var newResiduals = evaluate(parameters).Residuals;
                    double newCost = ComputeCost(newResiduals);

                    if (newCost < cost)
                    {
                        lambda = Math.Max(lambda / options.DampingDecreaseFactor, 1e-10);
                        accepted = true;
                    }
                    else
                    {
                        Array.Copy(originalParams, parameters, parameters.Length);
                        lambda = Math.Min(lambda * options.DampingIncreaseFactor, 1e10);
                        innerIterations++;
                    }
                }
                else
                {
                    double alpha = LineSearch(parameters, delta, p => evaluate(p).Residuals, cost, options.LineSearchSteps);

                    if (alpha == 0)
                    {
                        return CreateResult(false, iter, cost, "Line search failed", startTime);
                    }
                    accepted = true;
                }
            }

            if (!accepted)
            {
                return CreateResult(false, iter, cost, "Damping adjustment failed", startTime);
            }

            prevCost = cost;
        }

        double finalCost = ComputeCost(evaluate(parameters).Residuals);
        return CreateResult(false, options.MaxIterations, finalCost, "Max iterations reached", startTime);
    }

    private static readonly int VectorDoubleCount = Vector<double>.Count;

    private static double ComputeCost(double[] residuals)
    {
        return ComputeSumOfSquares(residuals);
    }

    private static double ComputeSumOfSquares(double[] v)
    {
        int n = v.Length;
        double sum = 0;
        int i = 0;

        if (n >= VectorDoubleCount)
        {
            var vSum = Vector<double>.Zero;
            int limit = n - VectorDoubleCount + 1;
            for (; i < limit; i += VectorDoubleCount)
            {
                var vec = new Vector<double>(v, i);
                vSum += vec * vec;
            }
            sum = Vector.Dot(vSum, Vector<double>.One);
        }

        for (; i < n; i++)
            sum += v[i] * v[i];
        return sum;
    }

    private static double[] SolveNormalEquations(double[,] J, double[] r, double lambda, bool useQR)
    {
        int m = J.GetLength(0);
        int n = J.GetLength(1);

        if (useQR)
        {
            if (lambda > 0)
            {
                // Augment system for regularization
                var augmentedJ = new double[m + n, n];
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        augmentedJ[i, j] = J[i, j];
                    }
                }
                double sqrtLambda = Math.Sqrt(lambda);
                for (int i = 0; i < n; i++)
                {
                    augmentedJ[m + i, i] = sqrtLambda;
                }

                var augmentedR = new double[m + n];
                for (int i = 0; i < m; i++)
                {
                    augmentedR[i] = -r[i];
                }
                // Rest are zeros

                return LinearSolver.QrSolve(augmentedJ, augmentedR);
            }
            else
            {
                var negR = new double[m];
                for (int i = 0; i < m; i++)
                {
                    negR[i] = -r[i];
                }
                return LinearSolver.QrSolve(J, negR);
            }
        }

        var JtJ = LinearSolver.ComputeJtJ(J);
        var Jtr = LinearSolver.ComputeJtr(J, r);
        var negJtr = new double[n];
        for (int i = 0; i < n; i++)
        {
            negJtr[i] = -Jtr[i];
        }

        if (lambda > 0)
        {
            for (int i = 0; i < n; i++)
            {
                JtJ[i, i] += lambda;
            }
        }

        try
        {
            return LinearSolver.CholeskySolve(JtJ, negJtr);
        }
        catch
        {
            if (lambda == 0)
            {
                double fallbackLambda = 1e-6;
                for (int i = 0; i < n; i++)
                {
                    JtJ[i, i] += fallbackLambda;
                }
                return LinearSolver.CholeskySolve(JtJ, negJtr);
            }
            throw;
        }
    }

    private static double[] SolveSparseNormalEquations(SparseMatrix J, double[] r, double lambda)
    {
        int n = J.Cols;

        var JtJ = J.ComputeJtJ();

        if (lambda > 0)
        {
            JtJ = JtJ.AddDiagonal(lambda);
        }

        var Jtr = J.ComputeJtr(r);
        var negJtr = new double[n];
        for (int i = 0; i < n; i++)
        {
            negJtr[i] = -Jtr[i];
        }

        return SparseLinearSolver.PreconditionedConjugateGradient(JtJ, negJtr);
    }

    private static double[] SolveSparseNormalEquationsDirect(SparseMatrix J, double[] r, double lambda)
    {
        int n = J.Cols;

        var JtJ = J.ComputeJtJ();
        if (lambda > 0) JtJ = JtJ.AddDiagonal(lambda);

        var Jtr = J.ComputeJtr(r);
        var negJtr = new double[n];
        for (int i = 0; i < n; i++) negJtr[i] = -Jtr[i];

        return SparseLinearSolver.SparseCholeskyDirect(JtJ, negJtr);
    }

    private static double LineSearch(
        double[] parameters,
        double[] delta,
        Func<double[], double[]> residualFn,
        double currentCost,
        int maxSteps)
    {
        var originalParams = (double[])parameters.Clone();
        double alpha = 1.0;

        for (int i = 0; i < maxSteps; i++)
        {
            for (int j = 0; j < parameters.Length; j++)
            {
                parameters[j] = originalParams[j] + alpha * delta[j];
            }

            var newResiduals = residualFn(parameters);
            double newCost = ComputeCost(newResiduals);

            if (newCost < currentCost)
            {
                return alpha;
            }

            alpha *= 0.5;
        }

        Array.Copy(originalParams, parameters, parameters.Length);
        return 0;
    }

    private static LeastSquaresResult CreateResult(bool success, int iterations, double cost, string reason, DateTime startTime)
    {
        return new LeastSquaresResult
        {
            Success = success,
            Iterations = iterations,
            FinalCost = cost,
            ConvergenceReason = reason,
            ComputationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private static string FormatNumber(double value)
    {
        if (value == 0) return "0";
        double abs = Math.Abs(value);
        if (abs >= 0.01) return value.ToString(abs >= 10 ? "F2" : "F3");
        if (abs >= 1e-6) return value.ToString("F6");
        if (abs < 1e-8) return "~0";
        return value.ToString("F9");
    }
}
