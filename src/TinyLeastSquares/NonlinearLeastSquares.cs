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
        options ??= new LeastSquaresOptions();
        bool useQR = options.UseQR;
        return SolveCore(
            initialParams,
            p => { var e = residualFn(p); return (e.Residuals, (object)e.Jacobian); },
            (jacobian, residuals) => LinearSolver.ComputeJtr((double[,])jacobian, residuals),
            (jacobian, residuals, jtr) => PrepareDenseNormal((double[,])jacobian, residuals, jtr, useQR),
            SolveDenseNormal,
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
            p => { var e = residualFn(p); return (e.Residuals, (object)e.Jacobian); },
            (jacobian, residuals) => ((SparseMatrix)jacobian).ComputeJtr(residuals),
            (jacobian, residuals, jtr) => PrepareSparseNormal((SparseMatrix)jacobian, jtr),
            SolveSparseNormalPcg,
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
            p => { var e = residualFn(p); return (e.Residuals, (object)e.Jacobian); },
            (jacobian, residuals) => ((SparseMatrix)jacobian).ComputeJtr(residuals),
            (jacobian, residuals, jtr) => PrepareSparseNormal((SparseMatrix)jacobian, jtr),
            SolveSparseNormalDirect,
            jacobian => { var j = (SparseMatrix)jacobian; return $", nnz={j.NonZeroCount}, sparsity={j.Sparsity:P1}"; },
            options);
    }

    private static LeastSquaresResult SolveCore(
        double[] initialParams,
        Func<double[], (double[] Residuals, object Jacobian)> evaluate,
        Func<object, double[], double[]> computeGradient,
        Func<object, double[], double[], object> prepareNormal,
        Func<object, double, double[]> solvePrepared,
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

            // Gradient (Jᵀr) is cheap and drives the convergence checks; compute it first so a
            // converging iteration can bail out *before* paying for the expensive normal-equations
            // build below.
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

            // Only now (we will actually take a step) build the normal equations (JᵀJ, reusing
            // the Jᵀr just computed). These are invariant across the inner damping loop — only λ
            // changes — so they are built once here and reused on every rejection/re-solve.
            var prepared = prepareNormal(jacobian, residuals, Jtr);

            double[]? delta;
            bool accepted = false;
            int innerIterations = 0;

            while (!accepted && innerIterations < options.MaxInnerIterations)
            {
                try
                {
                    delta = solvePrepared(prepared, options.AdaptiveDamping ? lambda : 0);
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

    // --- Prepared normal equations ------------------------------------------------
    // The normal equations (JᵀJ, Jᵀr) depend only on the current Jacobian and residual,
    // not on the LM damping λ. We build them once per outer iteration (Prepare*) and the
    // inner damping loop only re-applies λ and re-factorizes (Solve*). This avoids
    // recomputing JᵀJ (dense O(mn²); sparse triplet build + sort) on every step rejection,
    // and computes Jᵀr a single time (it also yields the gradient norm).

    /// <summary>Reusable dense normal equations for the Cholesky path (or raw J for QR).</summary>
    private sealed class DenseNormalEquations
    {
        public readonly bool UseQR;

        // Cholesky path: JᵀJ (mutated on the diagonal per solve), its original diagonal, and -Jᵀr.
        public readonly double[,]? JtJ;
        public readonly double[]? OrigDiag;
        public readonly double[]? NegJtr;

        // QR path: the augmented system is λ-dependent and cannot be cached, so keep J and r.
        public readonly double[,]? J;
        public readonly double[]? Residuals;

        public DenseNormalEquations(double[,] jtj, double[] origDiag, double[] negJtr)
        {
            UseQR = false; JtJ = jtj; OrigDiag = origDiag; NegJtr = negJtr;
        }

        public DenseNormalEquations(double[,] j, double[] residuals, bool _)
        {
            UseQR = true; J = j; Residuals = residuals;
        }
    }

    private static object PrepareDenseNormal(double[,] J, double[] r, double[] Jtr, bool useQR)
    {
        if (useQR)
            return new DenseNormalEquations(J, r, true);

        int n = J.GetLength(1);
        var JtJ = LinearSolver.ComputeJtJ(J);
        var origDiag = new double[n];
        var negJtr = new double[n];
        for (int i = 0; i < n; i++)
        {
            origDiag[i] = JtJ[i, i];
            negJtr[i] = -Jtr[i];
        }
        return new DenseNormalEquations(JtJ, origDiag, negJtr);
    }

    private static double[] SolveDenseNormal(object prepared, double lambda)
    {
        var ne = (DenseNormalEquations)prepared;

        if (ne.UseQR)
            return SolveDenseQr(ne.J!, ne.Residuals!, lambda);

        var JtJ = ne.JtJ!;
        var origDiag = ne.OrigDiag!;
        var negJtr = ne.NegJtr!;
        int n = origDiag.Length;

        // Restore the diagonal and re-apply the current damping (λ >= 0). CholeskySolve
        // reads JtJ and allocates a fresh L; it does not mutate JtJ off-diagonal or negJtr,
        // so both are safe to reuse across inner iterations.
        for (int i = 0; i < n; i++) JtJ[i, i] = origDiag[i] + lambda;

        try
        {
            return LinearSolver.CholeskySolve(JtJ, negJtr);
        }
        catch
        {
            if (lambda == 0)
            {
                for (int i = 0; i < n; i++) JtJ[i, i] = origDiag[i] + 1e-6;
                return LinearSolver.CholeskySolve(JtJ, negJtr);
            }
            throw;
        }
    }

    private static double[] SolveDenseQr(double[,] J, double[] r, double lambda)
    {
        int m = J.GetLength(0);
        int n = J.GetLength(1);

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

    /// <summary>Reusable sparse normal equations: JᵀJ (built once) and -Jᵀr.</summary>
    private sealed class SparseNormalEquations
    {
        public readonly SparseMatrix JtJ;
        public readonly double[] NegJtr;

        public SparseNormalEquations(SparseMatrix jtj, double[] negJtr)
        {
            JtJ = jtj; NegJtr = negJtr;
        }
    }

    private static object PrepareSparseNormal(SparseMatrix J, double[] Jtr)
    {
        var JtJ = J.ComputeJtJ();
        int n = J.Cols;
        var negJtr = new double[n];
        for (int i = 0; i < n; i++) negJtr[i] = -Jtr[i];

        return new SparseNormalEquations(JtJ, negJtr);
    }

    private static double[] SolveSparseNormalPcg(object prepared, double lambda)
    {
        var ne = (SparseNormalEquations)prepared;
        // AddDiagonal returns a new matrix (fast path clones only Values), leaving the cached
        // JᵀJ untouched so it can be re-damped on the next inner iteration.
        var A = lambda > 0 ? ne.JtJ.AddDiagonal(lambda) : ne.JtJ;
        return SparseLinearSolver.PreconditionedConjugateGradient(A, ne.NegJtr);
    }

    private static double[] SolveSparseNormalDirect(object prepared, double lambda)
    {
        var ne = (SparseNormalEquations)prepared;
        var A = lambda > 0 ? ne.JtJ.AddDiagonal(lambda) : ne.JtJ;
        return SparseLinearSolver.SparseCholeskyDirect(A, ne.NegJtr);
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
