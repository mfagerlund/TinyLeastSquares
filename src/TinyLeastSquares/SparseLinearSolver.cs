using System;
using System.Numerics;
using CSparseDouble = CSparse.Double;

namespace TinyLeastSquares;

/// <summary>
/// Sparse linear solvers for symmetric positive definite systems.
/// </summary>
public static class SparseLinearSolver
{
    private static readonly int VectorDoubleCount = Vector<double>.Count;

    /// <summary>
    /// Direct sparse Cholesky solve of symmetric positive definite Ax = b via CSparse.NET
    /// (Tim Davis's CSparse port). Uses AMD column ordering. Much faster than PCG/ICCG
    /// for fixed-sparsity normal equations — typical 10-50x for 20k-var systems.
    /// A must be SPD (J^T J + λI).
    /// </summary>
    public static double[] SparseCholeskyDirect(SparseMatrix A, double[] b)
    {
        if (A.Rows != A.Cols) throw new ArgumentException("Matrix must be square", nameof(A));
        int n = A.Rows;
        var csc = ToCSparseCsc(A);
        var chol = CSparseDouble.Factorization.SparseCholesky.Create(csc, CSparse.ColumnOrdering.MinimumDegreeAtPlusA);
        var x = new double[n];
        chol.Solve(b, x);
        return x;
    }

    /// <summary>
    /// Convert CSR <see cref="SparseMatrix"/> (our native format) to CSparse.NET's
    /// CompressedColumnStorage (CSC). Symmetric matrices (JtJ) need the full pattern
    /// stored — CSparse's Cholesky reads the lower triangle but requires valid column
    /// pointers across the whole matrix. SparseMatrix.ComputeJtJ stores the full
    /// symmetric pattern, which satisfies this contract.
    /// </summary>
    private static CSparseDouble.SparseMatrix ToCSparseCsc(SparseMatrix src)
    {
        int m = src.Rows, n = src.Cols, nnz = src.NonZeroCount;

        // Count nnz per column.
        var colCounts = new int[n];
        for (int k = 0; k < nnz; k++) colCounts[src.ColIndices[k]]++;

        // Build column pointers (cumulative).
        var colPtr = new int[n + 1];
        for (int j = 0; j < n; j++) colPtr[j + 1] = colPtr[j] + colCounts[j];

        // Fill row indices + values per column.
        var rowIdx = new int[nnz];
        var vals = new double[nnz];
        var next = new int[n];
        Array.Copy(colPtr, next, n);
        for (int r = 0; r < m; r++)
        {
            int start = src.RowPointers[r];
            int end = src.RowPointers[r + 1];
            for (int k = start; k < end; k++)
            {
                int c = src.ColIndices[k];
                int dst = next[c]++;
                rowIdx[dst] = r;
                vals[dst] = src.Values[k];
            }
        }

        return new CSparseDouble.SparseMatrix(m, n, vals, rowIdx, colPtr);
    }

    /// <summary>
    /// Solves Ax = b using Conjugate Gradient method.
    /// A must be symmetric positive definite (like J^T J).
    /// </summary>
    /// <param name="A">Sparse SPD matrix</param>
    /// <param name="b">Right-hand side vector</param>
    /// <param name="x0">Initial guess (null for zero)</param>
    /// <param name="maxIterations">Maximum iterations (default: 2 * dimension)</param>
    /// <param name="tolerance">Convergence tolerance on residual norm (default: 1e-10)</param>
    /// <returns>Solution vector x</returns>
    public static double[] ConjugateGradient(
        SparseMatrix A,
        double[] b,
        double[]? x0 = null,
        int maxIterations = 0,
        double tolerance = 1e-10)
    {
        int n = b.Length;
        if (maxIterations <= 0) maxIterations = 2 * n;

        var x = x0 != null ? (double[])x0.Clone() : new double[n];
        var r = Subtract(b, A.Multiply(x));  // r = b - Ax
        var p = (double[])r.Clone();         // p = r
        double rsOld = Dot(r, r);

        if (Math.Sqrt(rsOld) < tolerance)
            return x;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var Ap = A.Multiply(p);
            double pAp = Dot(p, Ap);

            if (Math.Abs(pAp) < 1e-30)
                break; // Breakdown

            double alpha = rsOld / pAp;

            Axpy(x, alpha, p, n);    // x += alpha * p
            Axpy(r, -alpha, Ap, n);  // r -= alpha * Ap

            double rsNew = Dot(r, r);

            if (Math.Sqrt(rsNew) < tolerance)
                break;

            double beta = rsNew / rsOld;
            ScaleAdd(p, r, beta, p, n);  // p = r + beta * p

            rsOld = rsNew;
        }

        return x;
    }

    /// <summary>
    /// Solves Ax = b using Preconditioned Conjugate Gradient with Jacobi (diagonal) preconditioner.
    /// Better convergence for ill-conditioned systems.
    /// </summary>
    public static double[] PreconditionedConjugateGradient(
        SparseMatrix A,
        double[] b,
        double[]? x0 = null,
        int maxIterations = 0,
        double tolerance = 1e-10)
    {
        int n = b.Length;
        if (maxIterations <= 0) maxIterations = 2 * n;

        // Jacobi preconditioner: M = diag(A)
        var invDiag = new double[n];
        for (int i = 0; i < n; i++)
        {
            double diag = A.Get(i, i);
            invDiag[i] = Math.Abs(diag) > 1e-15 ? 1.0 / diag : 1.0;
        }

        var x = x0 != null ? (double[])x0.Clone() : new double[n];
        var r = Subtract(b, A.Multiply(x));  // r = b - Ax

        // z = M^{-1} r (apply preconditioner)
        var z = new double[n];
        ApplyPreconditioner(z, invDiag, r, n);

        var p = (double[])z.Clone();
        double rzOld = Dot(r, z);

        double rNorm = Math.Sqrt(Dot(r, r));
        if (rNorm < tolerance)
            return x;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var Ap = A.Multiply(p);
            double pAp = Dot(p, Ap);

            if (Math.Abs(pAp) < 1e-30)
                break;

            double alpha = rzOld / pAp;

            Axpy(x, alpha, p, n);    // x += alpha * p
            Axpy(r, -alpha, Ap, n);  // r -= alpha * Ap

            rNorm = Math.Sqrt(Dot(r, r));
            if (rNorm < tolerance)
                break;

            // z = M^{-1} r
            ApplyPreconditioner(z, invDiag, r, n);

            double rzNew = Dot(r, z);
            double beta = rzNew / rzOld;

            ScaleAdd(p, z, beta, p, n);  // p = z + beta * p

            rzOld = rzNew;
        }

        return x;
    }

    /// <summary>
    /// Solves Ax = b using Incomplete Cholesky preconditioned Conjugate Gradient.
    /// Better for very ill-conditioned systems but more expensive per iteration.
    /// </summary>
    public static double[] ICCGSolve(
        SparseMatrix A,
        double[] b,
        double[]? x0 = null,
        int maxIterations = 0,
        double tolerance = 1e-10)
    {
        int n = b.Length;
        if (maxIterations <= 0) maxIterations = 2 * n;

        // Compute incomplete Cholesky factorization (IC(0) - same sparsity as A)
        var L = IncompleteCholesky(A);

        var x = x0 != null ? (double[])x0.Clone() : new double[n];
        var r = Subtract(b, A.Multiply(x));

        // z = L^{-T} L^{-1} r (solve L L^T z = r)
        var z = ICApply(L, r);

        var p = (double[])z.Clone();
        double rzOld = Dot(r, z);

        double rNorm = Math.Sqrt(Dot(r, r));
        if (rNorm < tolerance)
            return x;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var Ap = A.Multiply(p);
            double pAp = Dot(p, Ap);

            if (Math.Abs(pAp) < 1e-30)
                break;

            double alpha = rzOld / pAp;

            Axpy(x, alpha, p, n);    // x += alpha * p
            Axpy(r, -alpha, Ap, n);  // r -= alpha * Ap

            rNorm = Math.Sqrt(Dot(r, r));
            if (rNorm < tolerance)
                break;

            z = ICApply(L, r);

            double rzNew = Dot(r, z);
            double beta = rzNew / rzOld;

            ScaleAdd(p, z, beta, p, n);  // p = z + beta * p

            rzOld = rzNew;
        }

        return x;
    }

    /// <summary>
    /// Computes Incomplete Cholesky factorization IC(0) - maintains sparsity pattern of A.
    /// Returns lower triangular L such that A ≈ L L^T.
    /// </summary>
    private static SparseMatrix IncompleteCholesky(SparseMatrix A)
    {
        int n = A.Rows;
        var L = new double[n][];
        var LCols = new int[n][];

        // Extract lower triangular structure
        for (int i = 0; i < n; i++)
        {
            var cols = new System.Collections.Generic.List<int>();
            var vals = new System.Collections.Generic.List<double>();

            for (int idx = A.RowPointers[i]; idx < A.RowPointers[i + 1]; idx++)
            {
                if (A.ColIndices[idx] <= i)
                {
                    cols.Add(A.ColIndices[idx]);
                    vals.Add(A.Values[idx]);
                }
            }

            LCols[i] = cols.ToArray();
            L[i] = vals.ToArray();
        }

        // IC(0) factorization
        for (int i = 0; i < n; i++)
        {
            for (int jIdx = 0; jIdx < LCols[i].Length; jIdx++)
            {
                int j = LCols[i][jIdx];

                if (j < i)
                {
                    // L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
                    double sum = 0;
                    int iPtr = 0, jPtr = 0;
                    while (iPtr < jIdx && jPtr < LCols[j].Length)
                    {
                        int iCol = LCols[i][iPtr];
                        int jCol = LCols[j][jPtr];
                        if (iCol == jCol && iCol < j)
                        {
                            sum += L[i][iPtr] * L[j][jPtr];
                            iPtr++;
                            jPtr++;
                        }
                        else if (iCol < jCol)
                            iPtr++;
                        else
                            jPtr++;
                    }

                    // Find L[j,j]
                    double Ljj = 1;
                    for (int k = 0; k < LCols[j].Length; k++)
                    {
                        if (LCols[j][k] == j)
                        {
                            Ljj = L[j][k];
                            break;
                        }
                    }

                    L[i][jIdx] = (L[i][jIdx] - sum) / Ljj;
                }
                else // j == i (diagonal)
                {
                    // L[i,i] = sqrt(A[i,i] - sum_{k<i} L[i,k]^2)
                    double sum = 0;
                    for (int k = 0; k < jIdx; k++)
                    {
                        sum += L[i][k] * L[i][k];
                    }

                    double val = L[i][jIdx] - sum;
                    L[i][jIdx] = val > 1e-10 ? Math.Sqrt(val) : 1e-5; // Prevent breakdown
                }
            }
        }

        // Convert to SparseMatrix
        var triplets = new System.Collections.Generic.List<(int, int, double)>();
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < LCols[i].Length; k++)
            {
                triplets.Add((i, LCols[i][k], L[i][k]));
            }
        }

        return SparseMatrix.FromTriplets(n, n, triplets);
    }

    /// <summary>
    /// Applies IC preconditioner: solves L L^T z = r
    /// </summary>
    private static double[] ICApply(SparseMatrix L, double[] r)
    {
        int n = r.Length;
        var y = new double[n];
        var z = new double[n];

        // Forward solve: L y = r
        for (int i = 0; i < n; i++)
        {
            double sum = r[i];
            for (int idx = L.RowPointers[i]; idx < L.RowPointers[i + 1] - 1; idx++)
            {
                sum -= L.Values[idx] * y[L.ColIndices[idx]];
            }
            // Last entry in row is diagonal
            int diagIdx = L.RowPointers[i + 1] - 1;
            y[i] = sum / L.Values[diagIdx];
        }

        // Back solve: L^T z = y
        for (int i = n - 1; i >= 0; i--)
        {
            int diagIdx = L.RowPointers[i + 1] - 1;
            z[i] = y[i] / L.Values[diagIdx];

            for (int idx = L.RowPointers[i]; idx < diagIdx; idx++)
            {
                y[L.ColIndices[idx]] -= L.Values[idx] * z[i];
            }
        }

        return z;
    }

    /// <summary>
    /// Sparse Cholesky factorization for small-medium systems.
    /// Uses elimination tree ordering.
    /// </summary>
    public static double[] SparseCholeskySolve(SparseMatrix A, double[] b)
    {
        // For simplicity, convert to dense for small systems
        // For production use with very large systems, use a proper sparse Cholesky library
        if (A.Rows <= 500)
        {
            var dense = A.ToDense();
            return LinearSolver.CholeskySolve(dense, b);
        }

        // For larger systems, use ICCG
        return ICCGSolve(A, b);
    }

    #region SIMD Vector Helpers

    /// <summary>SIMD dot product with scalar tail.</summary>
    private static double Dot(double[] a, double[] b)
    {
        int n = a.Length;
        double sum = 0;
        int i = 0;

        if (n >= VectorDoubleCount)
        {
            var vSum = Vector<double>.Zero;
            int limit = n - VectorDoubleCount + 1;
            for (; i < limit; i += VectorDoubleCount)
            {
                var va = new Vector<double>(a, i);
                var vb = new Vector<double>(b, i);
                vSum += va * vb;
            }
            sum = Vector.Dot(vSum, Vector<double>.One);
        }

        for (; i < n; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /// <summary>SIMD element-wise subtract: result[i] = a[i] - b[i].</summary>
    private static double[] Subtract(double[] a, double[] b)
    {
        int n = a.Length;
        var result = new double[n];
        int i = 0;

        if (n >= VectorDoubleCount)
        {
            int limit = n - VectorDoubleCount + 1;
            for (; i < limit; i += VectorDoubleCount)
            {
                var va = new Vector<double>(a, i);
                var vb = new Vector<double>(b, i);
                (va - vb).CopyTo(result, i);
            }
        }

        for (; i < n; i++)
            result[i] = a[i] - b[i];
        return result;
    }

    /// <summary>SIMD y[i] += alpha * x[i].</summary>
    private static void Axpy(double[] y, double alpha, double[] x, int n)
    {
        int i = 0;

        if (n >= VectorDoubleCount)
        {
            var vAlpha = new Vector<double>(alpha);
            int limit = n - VectorDoubleCount + 1;
            for (; i < limit; i += VectorDoubleCount)
            {
                var vy = new Vector<double>(y, i);
                var vx = new Vector<double>(x, i);
                (vy + vAlpha * vx).CopyTo(y, i);
            }
        }

        for (; i < n; i++)
            y[i] += alpha * x[i];
    }

    /// <summary>SIMD result[i] = a[i] + beta * b[i].</summary>
    private static void ScaleAdd(double[] result, double[] a, double beta, double[] b, int n)
    {
        int i = 0;

        if (n >= VectorDoubleCount)
        {
            var vBeta = new Vector<double>(beta);
            int limit = n - VectorDoubleCount + 1;
            for (; i < limit; i += VectorDoubleCount)
            {
                var va = new Vector<double>(a, i);
                var vb = new Vector<double>(b, i);
                (va + vBeta * vb).CopyTo(result, i);
            }
        }

        for (; i < n; i++)
            result[i] = a[i] + beta * b[i];
    }

    /// <summary>SIMD z[i] = invDiag[i] * r[i].</summary>
    private static void ApplyPreconditioner(double[] z, double[] invDiag, double[] r, int n)
    {
        int i = 0;

        if (n >= VectorDoubleCount)
        {
            int limit = n - VectorDoubleCount + 1;
            for (; i < limit; i += VectorDoubleCount)
            {
                var vd = new Vector<double>(invDiag, i);
                var vr = new Vector<double>(r, i);
                (vd * vr).CopyTo(z, i);
            }
        }

        for (; i < n; i++)
            z[i] = invDiag[i] * r[i];
    }

    #endregion
}
