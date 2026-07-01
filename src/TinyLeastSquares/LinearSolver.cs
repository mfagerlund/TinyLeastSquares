using System;
using System.Numerics;

namespace TinyLeastSquares;

/// <summary>
/// Linear algebra utilities for solving linear systems.
/// Includes Cholesky decomposition, QR decomposition, and related solvers.
/// </summary>
public static class LinearSolver
{
    private static readonly int VectorDoubleCount = Vector<double>.Count;

    /// <summary>
    /// Performs Cholesky decomposition of a positive definite matrix A = LL^T.
    /// </summary>
    /// <param name="A">Positive definite matrix to decompose (n x n)</param>
    /// <returns>Lower triangular matrix L</returns>
    /// <exception cref="InvalidOperationException">If matrix is not positive definite</exception>
    public static double[,] CholeskyDecomposition(double[,] A)
    {
        int n = A.GetLength(0);
        var L = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double val = A[i, i] - sum;
                    if (val <= 0)
                    {
                        throw new InvalidOperationException(
                            $"Matrix is not positive definite at diagonal element {i}");
                    }
                    L[i, j] = Math.Sqrt(val);
                }
                else
                {
                    L[i, j] = (A[i, j] - sum) / L[j, j];
                }
            }
        }

        return L;
    }

    /// <summary>
    /// Solves Ly = b for y using forward substitution (L is lower triangular).
    /// </summary>
    public static double[] ForwardSubstitution(double[,] L, double[] b)
    {
        int n = L.GetLength(0);
        var y = new double[n];

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < i; j++)
            {
                sum += L[i, j] * y[j];
            }
            y[i] = (b[i] - sum) / L[i, i];
        }

        return y;
    }

    /// <summary>
    /// Solves L^T x = y for x using back substitution (L^T is upper triangular).
    /// </summary>
    public static double[] BackSubstitution(double[,] L, double[] y)
    {
        int n = L.GetLength(0);
        var x = new double[n];

        for (int i = n - 1; i >= 0; i--)
        {
            double sum = 0;
            for (int j = i + 1; j < n; j++)
            {
                sum += L[j, i] * x[j];
            }
            x[i] = (y[i] - sum) / L[i, i];
        }

        return x;
    }

    /// <summary>
    /// Solves Ax = b using Cholesky decomposition where A is positive definite.
    /// </summary>
    public static double[] CholeskySolve(double[,] A, double[] b)
    {
        var L = CholeskyDecomposition(A);
        var y = ForwardSubstitution(L, b);
        var x = BackSubstitution(L, y);
        return x;
    }

    /// <summary>
    /// Computes J^T J (Jacobian transpose times Jacobian).
    /// Uses column extraction for contiguous access and exploits symmetry.
    /// </summary>
    /// <param name="J">Jacobian matrix (m residuals x n parameters)</param>
    /// <returns>J^T J matrix (n x n)</returns>
    public static double[,] ComputeJtJ(double[,] J)
    {
        int m = J.GetLength(0);
        int n = J.GetLength(1);
        var JtJ = new double[n, n];

        // Extract columns into contiguous arrays for SIMD-friendly access
        var cols = new double[n][];
        for (int j = 0; j < n; j++)
        {
            var col = new double[m];
            for (int k = 0; k < m; k++)
                col[k] = J[k, j];
            cols[j] = col;
        }

        // Compute upper triangle using SIMD dot, then mirror
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double val = SimdDot(cols[i], cols[j], m);
                JtJ[i, j] = val;
                if (i != j) JtJ[j, i] = val;
            }
        }

        return JtJ;
    }

    /// <summary>
    /// Computes J^T r (Jacobian transpose times residual vector).
    /// </summary>
    /// <param name="J">Jacobian matrix (m residuals x n parameters)</param>
    /// <param name="r">Residual vector (m elements)</param>
    /// <returns>J^T r vector (n elements)</returns>
    public static double[] ComputeJtr(double[,] J, double[] r)
    {
        int m = J.GetLength(0);
        int n = J.GetLength(1);
        var Jtr = new double[n];

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int k = 0; k < m; k++)
                sum += J[k, i] * r[k];
            Jtr[i] = sum;
        }

        return Jtr;
    }

    /// <summary>
    /// Computes QR decomposition of matrix A using Householder reflections.
    /// Returns Q^T directly (the product of Householder reflectors applied to identity),
    /// which is the form needed by QrSolve to compute Q^T * b efficiently.
    /// </summary>
    /// <param name="A">Input matrix (m x n)</param>
    /// <returns>Tuple of Qt (Q^T, m x m) and R (m x n) matrices, where A = Q * R = Qt^T * R</returns>
    public static (double[,] Qt, double[,] R) QrDecomposition(double[,] A)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);

        // Copy A to R
        var R = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                R[i, j] = A[i, j];
            }
        }

        // Initialize Qt as identity; Householder reflectors accumulate H_k * ... * H_1 = Q^T
        var Qt = new double[m, m];
        for (int i = 0; i < m; i++)
        {
            Qt[i, i] = 1;
        }

        for (int k = 0; k < Math.Min(m - 1, n); k++)
        {
            double norm = 0;
            for (int i = k; i < m; i++)
            {
                norm += R[i, k] * R[i, k];
            }
            norm = Math.Sqrt(norm);

            if (norm < 1e-14) continue;

            double s = R[k, k] >= 0 ? -1 : 1;
            double u1 = R[k, k] - s * norm;
            var v = new double[m];
            v[k] = 1;
            for (int i = k + 1; i < m; i++)
            {
                v[i] = R[i, k] / u1;
            }

            double tau = -s * u1 / norm;

            // Update R
            for (int j = k; j < n; j++)
            {
                double sum = R[k, j];
                for (int i = k + 1; i < m; i++)
                {
                    sum += v[i] * R[i, j];
                }
                sum *= tau;

                R[k, j] -= sum;
                for (int i = k + 1; i < m; i++)
                {
                    R[i, j] -= sum * v[i];
                }
            }

            // Update Qt (accumulates Q^T = H_k * ... * H_1)
            for (int j = 0; j < m; j++)
            {
                double sum = Qt[k, j];
                for (int i = k + 1; i < m; i++)
                {
                    sum += v[i] * Qt[i, j];
                }
                sum *= tau;

                Qt[k, j] -= sum;
                for (int i = k + 1; i < m; i++)
                {
                    Qt[i, j] -= sum * v[i];
                }
            }
        }

        return (Qt, R);
    }

    /// <summary>
    /// Solves least squares problem min ||Ax - b|| using QR decomposition.
    /// Handles rank-deficient matrices by truncating small singular values.
    /// </summary>
    /// <param name="A">Coefficient matrix (m x n)</param>
    /// <param name="b">Right-hand side vector (m elements)</param>
    /// <param name="epsilon">Singular value threshold (default 1e-10)</param>
    /// <returns>Solution vector x (n elements)</returns>
    public static double[] QrSolve(double[,] A, double[] b, double epsilon = 1e-10)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);

        var (Qt, R) = QrDecomposition(A);

        // Compute Q^T * b = Qt * b
        var Qtb = new double[m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                Qtb[i] += Qt[i, j] * b[j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = Math.Min(m, n) - 1; i >= 0; i--)
        {
            if (Math.Abs(R[i, i]) < epsilon)
            {
                x[i] = 0;
                continue;
            }

            double sum = Qtb[i];
            for (int j = i + 1; j < n; j++)
            {
                sum -= R[i, j] * x[j];
            }
            x[i] = sum / R[i, i];
        }

        return x;
    }

    /// <summary>SIMD dot product for contiguous double arrays.</summary>
    private static double SimdDot(double[] a, double[] b, int n)
    {
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
}
