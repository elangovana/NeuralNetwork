using System;
using System.Linq;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    /// <summary>
    ///     Calculates Fitness by percentage correct
    /// </summary>
    public class ClassficationFitnessCalculator : IFitnessCalculator
    {
        public double Calculator(double[][] targetOutputs, double[][] actualOutputs)
        {
            int totalCorrect = targetOutputs.Where((t, r) => GetDigit(t) == GetDigit(actualOutputs[r])).Count();

            return (totalCorrect*100.0)/actualOutputs.Length;
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