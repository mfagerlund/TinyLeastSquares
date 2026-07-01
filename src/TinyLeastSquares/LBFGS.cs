using System;

namespace TinyLeastSquares;

/// <summary>
/// Configuration options for the L-BFGS solver.
/// </summary>
public class LBFGSOptions
{
    /// <summary>Maximum number of outer iterations.</summary>
    public int MaxIterations { get; set; } = 100;

    /// <summary>Stop when |E_prev - E_new| / max(|E_new|, 1) is below this.</summary>
    public double CostTolerance { get; set; } = 1e-9;

    /// <summary>Stop when ||delta||_inf is below this.</summary>
    public double ParamTolerance { get; set; } = 1e-9;

    /// <summary>Stop when ||gradient||_inf is below this.</summary>
    public double GradientTolerance { get; set; } = 1e-6;

    /// <summary>Number of stored {s, y} vector pairs (history depth). Reference uses 8.</summary>
    public int HistorySize { get; set; } = 8;

    /// <summary>Armijo sufficient-decrease constant. Typical 1e-4.</summary>
    public double ArmijoC1 { get; set; } = 1e-4;

    /// <summary>Backtracking step contraction factor.</summary>
    public double BacktrackContract { get; set; } = 0.5;

    /// <summary>Maximum backtracking steps before bailing out of line search.</summary>
    public int MaxLineSearchSteps { get; set; } = 30;

    /// <summary>Initial trial step length.</summary>
    public double InitialStep { get; set; } = 1.0;

    /// <summary>Enable verbose logging via callback.</summary>
    public bool Verbose { get; set; } = false;

    /// <summary>Optional callback for logging.</summary>
    public Action<string>? LogCallback { get; set; }
}

/// <summary>
/// Result object returned by the L-BFGS solver.
/// </summary>
public class LBFGSResult
{
    /// <summary>Whether the solver converged successfully.</summary>
    public bool Success { get; init; }

    /// <summary>Number of iterations performed.</summary>
    public int Iterations { get; init; }

    /// <summary>Final objective (energy) value.</summary>
    public double FinalCost { get; init; }

    /// <summary>Infinity norm of the final gradient.</summary>
    public double FinalGradientNorm { get; init; }

    /// <summary>Reason for convergence or termination.</summary>
    public string ConvergenceReason { get; init; } = "";

    /// <summary>Computation time in milliseconds.</summary>
    public double ComputationTimeMs { get; init; }
}

/// <summary>
/// Delegate for evaluating the scalar objective and its dense gradient.
/// </summary>
public delegate (double energy, double[] gradient) EnergyGradientFunction(double[] parameters);

/// <summary>
/// Limited-memory BFGS solver for unconstrained smooth minimization
/// of f: R^n -> R using only function value and gradient evaluations.
///
/// Algorithm: Nocedal &amp; Wright Ch. 7 two-loop recursion with backtracking
/// Armijo line search.
///
/// Use this when you have a scalar energy with an explicit gradient and no
/// natural least-squares decomposition. For Σ r_i² problems prefer
/// <see cref="NonlinearLeastSquaresSolver"/>.
/// </summary>
public static class LBFGSSolver
{
    /// <summary>
    /// Minimizes a scalar objective f: R^n -> R using L-BFGS.
    /// </summary>
    /// <param name="initialParams">Initial parameter values (modified in place).</param>
    /// <param name="evalFn">Function returning the objective value and its gradient.</param>
    /// <param name="options">Solver options (optional).</param>
    /// <returns>Result containing convergence info.</returns>
    public static LBFGSResult Solve(
        double[] initialParams,
        EnergyGradientFunction evalFn,
        LBFGSOptions? options = null)
    {
        options ??= new LBFGSOptions();
        var startTime = DateTime.UtcNow;

        int n = initialParams.Length;
        int m = Math.Max(1, options.HistorySize);

        var x = initialParams; // modified in-place
        var (f, g) = evalFn(x);

        void Log(string msg)
        {
            if (options.Verbose) options.LogCallback?.Invoke(msg);
        }

        double gInf = InfNorm(g);
        Log($"L-BFGS iter 0: f={f:G6}, ||g||_inf={gInf:G6}");

        if (gInf < options.GradientTolerance)
        {
            return Build(true, 0, f, gInf, "Gradient tolerance reached at start", startTime);
        }

        // History buffers (ring)
        var sHist = new double[m][];
        var yHist = new double[m][];
        var rhoHist = new double[m];
        int stored = 0;
        int head = 0; // index of next slot to write

        var q = new double[n];
        var alpha = new double[m];
        var direction = new double[n];
        var xNew = new double[n];
        var prevG = new double[n];

        for (int iter = 1; iter <= options.MaxIterations; iter++)
        {
            // --- Two-loop recursion to get H_k * g ---
            Array.Copy(g, q, n);

            // First loop: newest -> oldest
            for (int k = 0; k < stored; k++)
            {
                int idx = (head - 1 - k + m) % m;
                double a = rhoHist[idx] * Dot(sHist[idx], q);
                alpha[idx] = a;
                AxpyInPlace(q, yHist[idx], -a);
            }

            // Initial Hessian scaling H0 = (s^T y) / (y^T y) * I (Nocedal eq. 7.20)
            double gamma = 1.0;
            if (stored > 0)
            {
                int idxLast = (head - 1 + m) % m;
                double sy = Dot(sHist[idxLast], yHist[idxLast]);
                double yy = Dot(yHist[idxLast], yHist[idxLast]);
                if (yy > 0) gamma = sy / yy;
            }
            for (int i = 0; i < n; i++) q[i] *= gamma;

            // Second loop: oldest -> newest
            for (int k = stored - 1; k >= 0; k--)
            {
                int idx = (head - 1 - k + m) % m;
                double beta = rhoHist[idx] * Dot(yHist[idx], q);
                AxpyInPlace(q, sHist[idx], alpha[idx] - beta);
            }

            // Search direction p = -H g
            for (int i = 0; i < n; i++) direction[i] = -q[i];

            double dirDotG = Dot(direction, g);
            if (dirDotG >= 0)
            {
                // Not a descent direction (numerical issue or bad curvature). Reset to steepest descent.
                Log($"  iter {iter}: non-descent dir (d·g={dirDotG:G3}), resetting history");
                for (int i = 0; i < n; i++) direction[i] = -g[i];
                dirDotG = -Dot(g, g);
                stored = 0;
                head = 0;
            }

            // --- Backtracking Armijo line search ---
            double step = options.InitialStep;
            double fNew = 0;
            double[]? gNew = null;
            bool accepted = false;
            int ls;

            for (ls = 0; ls < options.MaxLineSearchSteps; ls++)
            {
                for (int i = 0; i < n; i++) xNew[i] = x[i] + step * direction[i];
                var (fTrial, gTrial) = evalFn(xNew);

                if (fTrial <= f + options.ArmijoC1 * step * dirDotG)
                {
                    fNew = fTrial;
                    gNew = gTrial;
                    accepted = true;
                    break;
                }
                step *= options.BacktrackContract;
            }

            if (!accepted)
            {
                Log($"  iter {iter}: line search failed after {ls} backtracks");
                return Build(false, iter, f, InfNorm(g), "Line search failed", startTime);
            }

            // Convergence check on step size
            double maxStep = 0;
            for (int i = 0; i < n; i++)
            {
                double d = Math.Abs(xNew[i] - x[i]);
                if (d > maxStep) maxStep = d;
            }

            // Update history: s = x_new - x, y = g_new - g
            var s = new double[n];
            var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                s[i] = xNew[i] - x[i];
                y[i] = gNew![i] - g[i];
            }
            double syNew = Dot(s, y);

            // Curvature condition: skip update if s^T y is not safely positive
            if (syNew > 1e-12)
            {
                sHist[head] = s;
                yHist[head] = y;
                rhoHist[head] = 1.0 / syNew;
                head = (head + 1) % m;
                if (stored < m) stored++;
            }
            else
            {
                Log($"  iter {iter}: skipping curvature update (s·y={syNew:G3})");
            }

            // Commit step
            Array.Copy(g, prevG, n);
            double prevF = f;
            Array.Copy(xNew, x, n);
            f = fNew;
            g = gNew!;
            gInf = InfNorm(g);

            Log($"L-BFGS iter {iter}: f={f:G6}, ||g||_inf={gInf:G6}, step={step:G3}, ls={ls + 1}");

            if (gInf < options.GradientTolerance)
                return Build(true, iter, f, gInf, "Gradient tolerance reached", startTime);

            double relCost = Math.Abs(prevF - f) / Math.Max(Math.Abs(f), 1.0);
            if (relCost < options.CostTolerance)
                return Build(true, iter, f, gInf, "Cost tolerance reached", startTime);

            if (maxStep < options.ParamTolerance)
                return Build(true, iter, f, gInf, "Parameter tolerance reached", startTime);
        }

        return Build(false, options.MaxIterations, f, gInf, "Max iterations reached", startTime);
    }

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static void AxpyInPlace(double[] y, double[] x, double a)
    {
        for (int i = 0; i < y.Length; i++) y[i] += a * x[i];
    }

    private static double InfNorm(double[] v)
    {
        double m = 0;
        for (int i = 0; i < v.Length; i++)
        {
            double a = Math.Abs(v[i]);
            if (a > m) m = a;
        }
        return m;
    }

    private static LBFGSResult Build(bool success, int iter, double f, double gNorm, string reason, DateTime t0)
    {
        return new LBFGSResult
        {
            Success = success,
            Iterations = iter,
            FinalCost = f,
            FinalGradientNorm = gNorm,
            ConvergenceReason = reason,
            ComputationTimeMs = (DateTime.UtcNow - t0).TotalMilliseconds
        };
    }
}
