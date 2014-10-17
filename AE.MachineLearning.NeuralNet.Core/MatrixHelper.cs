using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class MatrixHelper
    {
        public static double[][] CreateMatrix(int rows, int cols)
        {
            var result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        public static void PrintVector(double[] vector, int decimals, bool blankLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i%12 == 0) // max of 12 values per row 
                    Console.WriteLine("");
                if (vector[i] >= 0.0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " "); // n decimals
            }
            if (blankLine) Console.WriteLine("\n");
        }

        public static void PrintMatrix(double[][] matrix, int numRows, int decimals)
        {
            if (numRows == -1) numRows = int.MaxValue;

            for (int i = 0; i < matrix.Length && i < numRows; ++i)
            {
                for (int j = 0; j < matrix[0].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" ");
                    Console.Write(matrix[i][j].ToString("F" + decimals) + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}