using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TinyLeastSquares;

/// <summary>
/// Sparse matrix in CSR (Compressed Sparse Row) format.
/// Efficient for row operations, matrix-vector multiplication, and J^T J computation.
/// </summary>
public class SparseMatrix
{
    /// <summary>Row pointers - rowPtr[i] is the index into Values/ColIndices where row i starts.</summary>
    public int[] RowPointers { get; private set; }

    /// <summary>Column indices for each non-zero value.</summary>
    public int[] ColIndices { get; private set; }

    /// <summary>Non-zero values.</summary>
    public double[] Values { get; private set; }

    /// <summary>Number of rows.</summary>
    public int Rows { get; }

    /// <summary>Number of columns.</summary>
    public int Cols { get; }

    /// <summary>Number of non-zero entries.</summary>
    public int NonZeroCount => Values.Length;

    private SparseMatrix(int rows, int cols, int[] rowPointers, int[] colIndices, double[] values)
    {
        Rows = rows;
        Cols = cols;
        RowPointers = rowPointers;
        ColIndices = colIndices;
        Values = values;
    }

    /// <summary>
    /// Creates a sparse matrix from COO (coordinate) format triplets.
    /// </summary>
    public static SparseMatrix FromTriplets(int rows, int cols, List<(int row, int col, double value)> triplets)
    {
        // Sort by row, then by column
        triplets.Sort((a, b) =>
        {
            int rowCmp = a.row.CompareTo(b.row);
            return rowCmp != 0 ? rowCmp : a.col.CompareTo(b.col);
        });

        // Merge duplicates (sum values at same position)
        var merged = new List<(int row, int col, double value)>();
        foreach (var t in triplets)
        {
            if (merged.Count > 0 && merged[^1].row == t.row && merged[^1].col == t.col)
            {
                var last = merged[^1];
                merged[^1] = (last.row, last.col, last.value + t.value);
            }
            else if (Math.Abs(t.value) > 1e-15) // Skip near-zero values
            {
                merged.Add(t);
            }
        }

        int nnz = merged.Count;
        var rowPointers = new int[rows + 1];
        var colIndices = new int[nnz];
        var values = new double[nnz];

        int idx = 0;
        for (int row = 0; row < rows; row++)
        {
            rowPointers[row] = idx;
            while (idx < nnz && merged[idx].row == row)
            {
                colIndices[idx] = merged[idx].col;
                values[idx] = merged[idx].value;
                idx++;
            }
        }
        rowPointers[rows] = nnz;

        return new SparseMatrix(rows, cols, rowPointers, colIndices, values);
    }

    /// <summary>
    /// Creates a sparse matrix using a builder pattern for efficient row-by-row construction.
    /// </summary>
    public static Builder CreateBuilder(int rows, int cols, int estimatedNnzPerRow = 4)
    {
        return new Builder(rows, cols, estimatedNnzPerRow);
    }

    /// <summary>
    /// Gets a value at (row, col). Returns 0 if not present.
    /// O(log(nnz_in_row)) due to binary search.
    /// </summary>
    public double Get(int row, int col)
    {
        int start = RowPointers[row];
        int end = RowPointers[row + 1];

        // Binary search within the row
        while (start < end)
        {
            int mid = (start + end) / 2;
            if (ColIndices[mid] == col)
                return Values[mid];
            if (ColIndices[mid] < col)
                start = mid + 1;
            else
                end = mid;
        }
        return 0;
    }

    private const int ParallelThreshold = 1000;

    /// <summary>
    /// Multiplies this matrix by a vector: y = A * x
    /// Uses parallel execution for large matrices.
    /// </summary>
    public double[] Multiply(double[] x)
    {
        if (x.Length != Cols)
            throw new ArgumentException($"Vector length {x.Length} does not match matrix columns {Cols}");

        var y = new double[Rows];

        if (Rows >= ParallelThreshold)
        {
            Parallel.For(0, Rows, row =>
            {
                double sum = 0;
                for (int idx = RowPointers[row]; idx < RowPointers[row + 1]; idx++)
                {
                    sum += Values[idx] * x[ColIndices[idx]];
                }
                y[row] = sum;
            });
        }
        else
        {
            for (int row = 0; row < Rows; row++)
            {
                double sum = 0;
                for (int idx = RowPointers[row]; idx < RowPointers[row + 1]; idx++)
                {
                    sum += Values[idx] * x[ColIndices[idx]];
                }
                y[row] = sum;
            }
        }

        return y;
    }

    /// <summary>
    /// Multiplies this matrix transpose by a vector: y = A^T * x
    /// </summary>
    public double[] TransposeMultiply(double[] x)
    {
        if (x.Length != Rows)
            throw new ArgumentException($"Vector length {x.Length} does not match matrix rows {Rows}");

        var y = new double[Cols];
        for (int row = 0; row < Rows; row++)
        {
            double xRow = x[row];
            for (int idx = RowPointers[row]; idx < RowPointers[row + 1]; idx++)
            {
                y[ColIndices[idx]] += Values[idx] * xRow;
            }
        }
        return y;
    }

    /// <summary>
    /// Computes J^T * J where this matrix is J.
    /// Returns a sparse symmetric matrix. Uses parallel execution for large matrices.
    /// </summary>
    public SparseMatrix ComputeJtJ()
    {
        if (Rows >= ParallelThreshold)
            return ComputeJtJParallel();

        var triplets = new List<(int row, int col, double value)>();

        for (int k = 0; k < Rows; k++)
        {
            int rowStart = RowPointers[k];
            int rowEnd = RowPointers[k + 1];

            for (int idx1 = rowStart; idx1 < rowEnd; idx1++)
            {
                int i = ColIndices[idx1];
                double vi = Values[idx1];

                for (int idx2 = idx1; idx2 < rowEnd; idx2++)
                {
                    int j = ColIndices[idx2];
                    double vj = Values[idx2];

                    double contrib = vi * vj;
                    triplets.Add((i, j, contrib));
                    if (i != j)
                    {
                        triplets.Add((j, i, contrib));
                    }
                }
            }
        }

        return FromTriplets(Cols, Cols, triplets);
    }

    private SparseMatrix ComputeJtJParallel()
    {
        var threadLocal = new ThreadLocal<List<(int row, int col, double value)>>(
            () => new List<(int, int, double)>(), trackAllValues: true);

        Parallel.For(0, Rows, k =>
        {
            var local = threadLocal.Value!;
            int rowStart = RowPointers[k];
            int rowEnd = RowPointers[k + 1];

            for (int idx1 = rowStart; idx1 < rowEnd; idx1++)
            {
                int i = ColIndices[idx1];
                double vi = Values[idx1];

                for (int idx2 = idx1; idx2 < rowEnd; idx2++)
                {
                    int j = ColIndices[idx2];
                    double vj = Values[idx2];

                    double contrib = vi * vj;
                    local.Add((i, j, contrib));
                    if (i != j)
                    {
                        local.Add((j, i, contrib));
                    }
                }
            }
        });

        // Merge all thread-local lists
        var triplets = new List<(int row, int col, double value)>();
        foreach (var list in threadLocal.Values)
            triplets.AddRange(list);
        threadLocal.Dispose();

        return FromTriplets(Cols, Cols, triplets);
    }

    /// <summary>
    /// Computes J^T * r where this matrix is J and r is the residual vector.
    /// </summary>
    public double[] ComputeJtr(double[] r)
    {
        return TransposeMultiply(r);
    }

    /// <summary>
    /// Adds lambda to the diagonal (for Levenberg-Marquardt damping).
    /// Returns a new matrix. Uses direct CSR manipulation instead of triplet rebuild.
    /// </summary>
    public SparseMatrix AddDiagonal(double lambda)
    {
        int diag = Math.Min(Rows, Cols);

        // Check if all diagonal entries already exist (fast path)
        bool allDiagonalsPresent = true;
        for (int i = 0; i < diag; i++)
        {
            if (FindIndex(i, i) < 0)
            {
                allDiagonalsPresent = false;
                break;
            }
        }

        if (allDiagonalsPresent)
        {
            // Fast path: clone Values, share RowPointers/ColIndices (immutable structure)
            var newValues = (double[])Values.Clone();
            for (int i = 0; i < diag; i++)
            {
                newValues[FindIndex(i, i)] += lambda;
            }
            return new SparseMatrix(Rows, Cols, RowPointers, ColIndices, newValues);
        }

        // Slow path: single-pass CSR rebuild inserting missing diagonal entries
        int extraCount = 0;
        for (int i = 0; i < diag; i++)
        {
            if (FindIndex(i, i) < 0) extraCount++;
        }

        int newNnz = Values.Length + extraCount;
        var newRowPointers = new int[Rows + 1];
        var newColIndices = new int[newNnz];
        var newVals = new double[newNnz];

        int dest = 0;
        for (int row = 0; row < Rows; row++)
        {
            newRowPointers[row] = dest;
            int start = RowPointers[row];
            int end = RowPointers[row + 1];
            bool needDiag = row < diag && FindIndex(row, row) < 0;
            bool diagInserted = false;

            for (int idx = start; idx < end; idx++)
            {
                int col = ColIndices[idx];
                // Insert diagonal before this column if needed
                if (needDiag && !diagInserted && col > row)
                {
                    newColIndices[dest] = row;
                    newVals[dest] = lambda;
                    dest++;
                    diagInserted = true;
                }

                newColIndices[dest] = col;
                newVals[dest] = Values[idx] + (col == row ? lambda : 0);
                dest++;
            }

            // Diagonal goes at end of row if not yet inserted
            if (needDiag && !diagInserted)
            {
                newColIndices[dest] = row;
                newVals[dest] = lambda;
                dest++;
            }
        }
        newRowPointers[Rows] = dest;

        return new SparseMatrix(Rows, Cols, newRowPointers, newColIndices, newVals);
    }

    /// <summary>
    /// Binary search for the index of (row, col) in CSR arrays. Returns -1 if not found.
    /// </summary>
    private int FindIndex(int row, int col)
    {
        int start = RowPointers[row];
        int end = RowPointers[row + 1];
        while (start < end)
        {
            int mid = (start + end) / 2;
            if (ColIndices[mid] == col) return mid;
            if (ColIndices[mid] < col) start = mid + 1;
            else end = mid;
        }
        return -1;
    }

    /// <summary>
    /// Converts to dense matrix (for debugging/small matrices).
    /// </summary>
    public double[,] ToDense()
    {
        var dense = new double[Rows, Cols];
        for (int row = 0; row < Rows; row++)
        {
            for (int idx = RowPointers[row]; idx < RowPointers[row + 1]; idx++)
            {
                dense[row, ColIndices[idx]] = Values[idx];
            }
        }
        return dense;
    }

    /// <summary>
    /// Gets the sparsity ratio (fraction of zero entries).
    /// </summary>
    public double Sparsity => 1.0 - (double)NonZeroCount / (Rows * Cols);

    /// <summary>
    /// Builder for efficient row-by-row sparse matrix construction.
    /// </summary>
    public class Builder
    {
        private readonly int _rows;
        private readonly int _cols;
        private readonly List<int> _rowPointers;
        private readonly List<int> _colIndices;
        private readonly List<double> _values;
        private int _currentRow;

        internal Builder(int rows, int cols, int estimatedNnzPerRow)
        {
            _rows = rows;
            _cols = cols;
            _rowPointers = new List<int>(rows + 1) { 0 };
            _colIndices = new List<int>(rows * estimatedNnzPerRow);
            _values = new List<double>(rows * estimatedNnzPerRow);
            _currentRow = 0;
        }

        /// <summary>
        /// Starts a new row. Must be called in order from row 0 to rows-1.
        /// </summary>
        public void BeginRow(int row)
        {
            if (row != _currentRow)
                throw new InvalidOperationException($"Expected row {_currentRow}, got {row}");
        }

        /// <summary>
        /// Adds a non-zero entry to the current row.
        /// Columns must be added in ascending order within each row.
        /// </summary>
        public void Add(int col, double value)
        {
            if (Math.Abs(value) < 1e-15) return; // Skip near-zeros

            if (_colIndices.Count > _rowPointers[_currentRow] && col <= _colIndices[^1])
                throw new InvalidOperationException("Columns must be added in ascending order");

            _colIndices.Add(col);
            _values.Add(value);
        }

        /// <summary>
        /// Ends the current row and moves to the next.
        /// </summary>
        public void EndRow()
        {
            _currentRow++;
            _rowPointers.Add(_colIndices.Count);
        }

        /// <summary>
        /// Adds a complete row of entries (columns and values must be same length).
        /// Columns must be in ascending order.
        /// </summary>
        public void AddRow(int[] cols, double[] vals)
        {
            BeginRow(_currentRow);
            for (int i = 0; i < cols.Length; i++)
            {
                Add(cols[i], vals[i]);
            }
            EndRow();
        }

        /// <summary>
        /// Builds the final sparse matrix.
        /// </summary>
        public SparseMatrix Build()
        {
            // Fill remaining rows if not all were added
            while (_currentRow < _rows)
            {
                _rowPointers.Add(_colIndices.Count);
                _currentRow++;
            }

            return new SparseMatrix(
                _rows, _cols,
                _rowPointers.ToArray(),
                _colIndices.ToArray(),
                _values.ToArray());
        }
    }
}
