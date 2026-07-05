using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using TinyLeastSquares;

if (args.Length > 0 && args[0] == "diag")
{
    SolverDiagnostics.Run();
    return;
}

BenchmarkRunner.Run<SolverBenchmarks>();

/// <summary>
/// Representative Levenberg-Marquardt workloads for TinyLeastSquares.
///
/// Each benchmark re-runs a full solve from a fixed start so the measured cost includes
/// the whole outer/inner iteration structure (residual + Jacobian eval, JtJ/Jtr, factorize,
/// damping accept/reject). MemoryDiagnoser reports per-solve allocations, which matter for the
/// small-repeated (per-frame IK) use case.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]  // warmup+iterations kept short so the whole suite runs in a couple of minutes; numbers are directional
public class SolverBenchmarks
{
    // ---------------------------------------------------------------------
    // 1. Exponential decay curve fit:  y = a*exp(-b*x) + c   (analytic Jacobian)
    //    n = 3 params, m = 256 data points. Mild nonlinearity, a few iterations.
    // ---------------------------------------------------------------------
    private double[] _expX = null!, _expY = null!;

    // ---------------------------------------------------------------------
    // 2. Gaussian-mixture fit: sum of K Gaussians (analytic Jacobian).
    //    n = 3K params, m = 500 samples. Strongly nonlinear -> LM rejects steps
    //    and the inner damping loop iterates. This is the case that stresses the
    //    per-inner-iteration JtJ/Jtr recomputation.
    // ---------------------------------------------------------------------
    private const int K = 6;                 // gaussians -> n = 18
    private double[] _gmX = null!, _gmY = null!;
    private double[] _gmTruth = null!, _gmStart = null!;

    // ---------------------------------------------------------------------
    // 3. Inverse kinematics: 6-link planar arm reaching a target (analytic Jacobian).
    //    n = 6 joint angles, m = 2 residuals. Tiny per-iteration cost -> dominated by
    //    allocation / iteration overhead. Represents the per-frame IK use case.
    // ---------------------------------------------------------------------
    private double[] _ikLen = null!;
    private double _ikBaseX, _ikBaseY;

    // ---------------------------------------------------------------------
    // 4. Sparse 2D distance-constraint chain solved with SparseSolveDirect.
    //    N particles in 2D (n = 2N), consecutive edge-length constraints + pinned ends.
    //    Nonlinear -> several iterations; each iteration builds JtJ and does a sparse
    //    Cholesky. Baseline for the sparse path.
    // ---------------------------------------------------------------------
    private const int ChainN = 300;          // n = 600
    private double _chainRest;
    private double[] _chainStart = null!;
    private (double x, double y) _chainA, _chainB;

    [GlobalSetup]
    public void Setup()
    {
        // --- exp decay ---
        var rng = new Random(1234);
        int m = 256;
        _expX = new double[m];
        _expY = new double[m];
        for (int i = 0; i < m; i++)
        {
            double x = i / (double)(m - 1) * 10.0;
            _expX[i] = x;
            _expY[i] = 70.0 * Math.Exp(-0.35 * x) + 15.0 + (rng.NextDouble() - 0.5) * 1.5;
        }

        // --- gaussian mixture ---
        _gmTruth = new double[3 * K];
        for (int k = 0; k < K; k++)
        {
            _gmTruth[3 * k + 0] = 1.0 + k * 0.3;              // amplitude
            _gmTruth[3 * k + 1] = -8.0 + 16.0 * k / (K - 1);  // center spread over [-8, 8]
            _gmTruth[3 * k + 2] = 0.8 + 0.15 * k;             // width
        }
        int gm = 500;
        _gmX = new double[gm];
        _gmY = new double[gm];
        for (int i = 0; i < gm; i++)
        {
            double x = -10.0 + 20.0 * i / (gm - 1);
            _gmX[i] = x;
            _gmY[i] = GmModel(_gmTruth, x) + (rng.NextDouble() - 0.5) * 0.02;
        }
        // Start perturbed from truth so LM has real work (and rejections) to do.
        _gmStart = (double[])_gmTruth.Clone();
        for (int j = 0; j < _gmStart.Length; j++)
            _gmStart[j] += (rng.NextDouble() - 0.5) * 0.6;

        // --- IK ---
        _ikLen = new[] { 52.0, 44.0, 34.0, 26.0, 20.0, 16.0 };
        _ikBaseX = 0; _ikBaseY = 0;

        // --- sparse chain ---
        _chainA = (0, 0);
        _chainB = (ChainN, 0);
        // rest length slightly longer than the straight chord so the chain must bow.
        _chainRest = 1.15;
        _chainStart = new double[2 * ChainN];
        var crng = new Random(7);
        for (int i = 0; i < ChainN; i++)
        {
            double t = i / (double)(ChainN - 1);
            _chainStart[2 * i + 0] = _chainA.x + (_chainB.x - _chainA.x) * t;
            _chainStart[2 * i + 1] = _chainA.y + (_chainB.y - _chainA.y) * t + (crng.NextDouble() - 0.5) * 2.0;
        }
    }

    // ============================ 1. Exp decay ============================
    [Benchmark]
    public double ExpDecayFit()
    {
        var p = new double[] { 30.0, 1.0, 0.0 };
        NonlinearLeastSquaresSolver.Solve(p, ExpEval, new LeastSquaresOptions { MaxIterations = 60 });
        return p[0] + p[1] + p[2];
    }

    private ResidualEvaluation ExpEval(double[] p)
    {
        double a = p[0], b = p[1], c = p[2];
        int m = _expX.Length;
        var r = new double[m];
        var j = new double[m, 3];
        for (int i = 0; i < m; i++)
        {
            double x = _expX[i];
            double e = Math.Exp(-b * x);
            r[i] = a * e + c - _expY[i];
            j[i, 0] = e;
            j[i, 1] = -a * x * e;
            j[i, 2] = 1.0;
        }
        return new ResidualEvaluation(r, j);
    }

    // ======================= 2. Gaussian mixture =======================
    [Benchmark]
    public double GaussianMixtureFit()
    {
        var p = (double[])_gmStart.Clone();
        NonlinearLeastSquaresSolver.Solve(p, GmEval, new LeastSquaresOptions { MaxIterations = 100 });
        double s = 0;
        for (int i = 0; i < p.Length; i++) s += p[i];
        return s;
    }

    private static double GmModel(double[] p, double x)
    {
        double sum = 0;
        for (int k = 0; k < K; k++)
        {
            double amp = p[3 * k + 0], mu = p[3 * k + 1], sig = p[3 * k + 2];
            double z = (x - mu) / sig;
            sum += amp * Math.Exp(-z * z);
        }
        return sum;
    }

    private ResidualEvaluation GmEval(double[] p)
    {
        int m = _gmX.Length;
        var r = new double[m];
        var j = new double[m, 3 * K];
        for (int i = 0; i < m; i++)
        {
            double x = _gmX[i];
            double f = 0;
            for (int k = 0; k < K; k++)
            {
                double amp = p[3 * k + 0], mu = p[3 * k + 1], sig = p[3 * k + 2];
                double z = (x - mu) / sig;
                double g = Math.Exp(-z * z);
                f += amp * g;
                j[i, 3 * k + 0] = g;                       // d/d amp
                j[i, 3 * k + 1] = amp * g * (2.0 * z / sig);   // d/d mu
                j[i, 3 * k + 2] = amp * g * (2.0 * z * z / sig); // d/d sigma
            }
            r[i] = f - _gmY[i];
        }
        return new ResidualEvaluation(r, j);
    }

    // ============================ 3. IK ============================
    [Benchmark]
    public double IkReach()
    {
        var theta = new double[_ikLen.Length]; // all zero start
        // A target that is reachable but forces the arm to fold.
        double tx = 60, ty = 40;
        NonlinearLeastSquaresSolver.Solve(theta, p => IkEval(p, tx, ty),
            new LeastSquaresOptions { MaxIterations = 60 });
        return theta[0];
    }

    private ResidualEvaluation IkEval(double[] theta, double tx, double ty)
    {
        int nJoints = theta.Length;
        // forward kinematics: cumulative angle, joint positions
        var px = new double[nJoints + 1];
        var py = new double[nJoints + 1];
        px[0] = _ikBaseX; py[0] = _ikBaseY;
        double ang = 0;
        for (int k = 0; k < nJoints; k++)
        {
            ang += theta[k];
            px[k + 1] = px[k] + _ikLen[k] * Math.Cos(ang);
            py[k + 1] = py[k] + _ikLen[k] * Math.Sin(ang);
        }
        double ex = px[nJoints], ey = py[nJoints];

        var r = new[] { ex - tx, ey - ty };
        var j = new double[2, nJoints];
        // rotating joint k spins the tip about joint-k position p_k
        for (int k = 0; k < nJoints; k++)
        {
            double dx = ex - px[k];
            double dy = ey - py[k];
            j[0, k] = -dy;
            j[1, k] = dx;
        }
        return new ResidualEvaluation(r, j);
    }

    // ======================= 4. Sparse chain =======================
    [Benchmark]
    public double SparseChainDirect()
    {
        var p = (double[])_chainStart.Clone();
        NonlinearLeastSquaresSolver.SparseSolveDirect(p, ChainEval,
            new LeastSquaresOptions { MaxIterations = 40 });
        return p[2]; // some interior coordinate
    }

    private SparseResidualEvaluation ChainEval(double[] p)
    {
        int edges = ChainN - 1;
        int n = 2 * ChainN;
        // rows: 2 pin-A + edges + 2 pin-B + n weak positional regularizers
        int m = edges + 4 + n;
        var r = new double[m];
        var builder = SparseMatrix.CreateBuilder(m, n, 4);

        int row = 0;
        double wAnchor = 100.0;   // stiff pins
        double wReg = 0.05;       // weak "stay near start" regularizer -> full-rank, well-conditioned

        // pin p0 -> A  (rows 0,1)
        builder.BeginRow(row); builder.Add(0, wAnchor); builder.EndRow();
        r[row++] = wAnchor * (p[0] - _chainA.x);
        builder.BeginRow(row); builder.Add(1, wAnchor); builder.EndRow();
        r[row++] = wAnchor * (p[1] - _chainA.y);

        // edge length constraints
        for (int e = 0; e < edges; e++)
        {
            int a = e, b = e + 1;
            double ax = p[2 * a], ay = p[2 * a + 1];
            double bx = p[2 * b], by = p[2 * b + 1];
            double dx = bx - ax, dy = by - ay;
            double dist = Math.Sqrt(dx * dx + dy * dy);
            if (dist < 1e-12) dist = 1e-12;
            r[row] = dist - _chainRest;

            // d dist / d(ax,ay,bx,by) = (-dx,-dy, dx, dy)/dist
            double idist = 1.0 / dist;
            // columns must be added in ascending order (b == a+1 so 2a < 2a+1 < 2b < 2b+1)
            builder.BeginRow(row);
            builder.Add(2 * a, -dx * idist);
            builder.Add(2 * a + 1, -dy * idist);
            builder.Add(2 * b, dx * idist);
            builder.Add(2 * b + 1, dy * idist);
            builder.EndRow();
            row++;
        }

        // pin p_{N-1} -> B
        int last = ChainN - 1;
        builder.BeginRow(row); builder.Add(2 * last, wAnchor); builder.EndRow();
        r[row++] = wAnchor * (p[2 * last] - _chainB.x);
        builder.BeginRow(row); builder.Add(2 * last + 1, wAnchor); builder.EndRow();
        r[row++] = wAnchor * (p[2 * last + 1] - _chainB.y);

        // weak positional regularizers: one residual per coordinate, single non-zero on the diagonal
        for (int c = 0; c < n; c++)
        {
            builder.BeginRow(row);
            builder.Add(c, wReg);
            builder.EndRow();
            r[row++] = wReg * (p[c] - _chainStart[c]);
        }

        return new SparseResidualEvaluation(r, builder.Build());
    }

    // ---- Diagnostic runners: count residual/Jacobian evaluations + iterations ----

    public (int iters, int evals, double cost) DiagExpDecay()
    {
        int calls = 0;
        var p = new double[] { 30.0, 1.0, 0.0 };
        var res = NonlinearLeastSquaresSolver.Solve(p, x => { calls++; return ExpEval(x); },
            new LeastSquaresOptions { MaxIterations = 60 });
        return (res.Iterations, calls, res.FinalCost);
    }

    public (int iters, int evals, double cost) DiagGaussianMixture()
    {
        int calls = 0;
        var p = (double[])_gmStart.Clone();
        var res = NonlinearLeastSquaresSolver.Solve(p, x => { calls++; return GmEval(x); },
            new LeastSquaresOptions { MaxIterations = 100 });
        return (res.Iterations, calls, res.FinalCost);
    }

    public (int iters, int evals, double cost) DiagIk()
    {
        int calls = 0;
        var theta = new double[_ikLen.Length];
        var res = NonlinearLeastSquaresSolver.Solve(theta, p => { calls++; return IkEval(p, 60, 40); },
            new LeastSquaresOptions { MaxIterations = 60 });
        return (res.Iterations, calls, res.FinalCost);
    }

    public (int iters, int evals, double cost) DiagSparseChain()
    {
        int calls = 0;
        var p = (double[])_chainStart.Clone();
        var res = NonlinearLeastSquaresSolver.SparseSolveDirect(p, x => { calls++; return ChainEval(x); },
            new LeastSquaresOptions { MaxIterations = 40 });
        return (res.Iterations, calls, res.FinalCost);
    }

}

public static class SolverDiagnostics
{
    public static void Run()
    {
        var b = new SolverBenchmarks();
        b.Setup();

        Console.WriteLine("Problem            | outerIters | evals(res+jac) | finalCost");
        Console.WriteLine("-------------------|------------|----------------|-----------");
        Print("ExpDecayFit", b.DiagExpDecay());
        Print("GaussianMixtureFit", b.DiagGaussianMixture());
        Print("IkReach", b.DiagIk());
        Print("SparseChainDirect", b.DiagSparseChain());

        Console.WriteLine();
        Console.WriteLine("Deterministic allocation per solve (GC.GetAllocatedBytesForCurrentThread):");
        Console.WriteLine("Problem            | KB/solve");
        Console.WriteLine("-------------------|---------");
        PrintAlloc("ExpDecayFit", () => b.ExpDecayFit(), 200);
        PrintAlloc("GaussianMixtureFit", () => b.GaussianMixtureFit(), 200);
        PrintAlloc("IkReach", () => b.IkReach(), 500);
        PrintAlloc("SparseChainDirect", () => b.SparseChainDirect(), 30);
    }

    private static void Print(string name, (int iters, int evals, double cost) d)
        => Console.WriteLine($"{name,-18} | {d.iters,10} | {d.evals,14} | {d.cost:G4}");

    private static void PrintAlloc(string name, Action solve, int reps)
    {
        solve(); // warm + JIT
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < reps; i++) solve();
        long after = GC.GetAllocatedBytesForCurrentThread();
        double kb = (after - before) / (double)reps / 1024.0;
        Console.WriteLine($"{name,-18} | {kb,8:F1}");
    }
}
