using System;
using System.Linq;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class ClassificationErrorCalculator : IErrorCalculator
    {
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