using System;
using System.Linq;

namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Calculates percentage correct
    /// </summary>
    public class ClassificationErrorCalculator : IErrorCalculator
    {
        /// <summary>
        /// Calcuates the error rates by dividing the total incorrect classifications with the total no of items to classify..
        /// </summary>
        /// <param name="targetOutputs">Expected outouts</param>
        /// <param name="actualOutputs">Actual outputs</param>
        /// <returns>Returns a value between 1 and 0, 0 indicting no errors, and 1 indicates 100% error</returns>
        public double CalculateError(double[][] targetOutputs, double[][] actualOutputs)
        {
            int totalCorrect = targetOutputs.Where((t, r) => GetDigit(t) == GetDigit(actualOutputs[r])).Count();

            return (1.0 - (totalCorrect)/(double) actualOutputs.Length);
        }

        private static int GetDigit(double[] entry)
        {
            for (int v = 0; v < entry.Length; v++)
            {
                if (Math.Round(entry[v], 4) == Math.Round(entry.Max(), 4)) return v;
            }

            return -1;
        }
    }
}