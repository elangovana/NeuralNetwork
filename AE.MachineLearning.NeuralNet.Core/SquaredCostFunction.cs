using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class SquaredCostFunction : ICostFunction
    {
        /// <summary>
        ///     Cost (Error per training data) -> [(t-a)^2]/2
        /// </summary>
        /// <param name="target">expected value of output</param>
        /// <param name="actual">actual Value or output</param>
        /// <returns>Cost -> (t-y)^2</returns>
        public double Cost(double target, double actual)
        {
            return Math.Pow(target - actual, 2)/2;
        }

        /// <summary>
        ///     Cost per Epoch or batch of training data {sum[(ti-ai)^2]}/2
        /// </summary>
        /// <param name="target">An array expected value</param>
        /// <param name="actual">An array of expected values. The array length much match that of the target array parameter</param>
        /// <returns>Cost</returns>
        public double Cost(double[] target, double[] actual)
        {
            if (target.Length != actual.Length)
                throw new NeuralNetException(
                    string.Format("The length {0} of the target array must match the length {1} of the acutal array",
                                  target.Length, actual.Length));

            double squaredSumOfError = 0.0;
            for (int i = 0; i < target.Length; i++)
            {
                squaredSumOfError = squaredSumOfError + Math.Pow(target[i] - actual[i], 2);
            }

            return squaredSumOfError/2;
        }

        /// <summary>
        ///     Partial Derivative of the cost function
        /// </summary>
        /// <param name="target">Expected output</param>
        /// <param name="actual">Acutual outoput</param>
        /// <returns>Derivative</returns>
        public double DerivativeCostWrtOutput(double target, double actual)
        {
            return (target - actual);
        }


        public double DerivativeCostWrtOutput(double[] target, double[] actual)
        {
            if (target.Length != actual.Length)
                throw new NeuralNetException(
                    string.Format("The length {0} of the target array must match the length {1} of the acutal array",
                                  target.Length, actual.Length));

            double sumOfError = 0.0;
            for (int i = 0; i < target.Length; i++)
            {
                sumOfError = sumOfError + actual[i] - target[i];
            }

            return sumOfError;
        }
    }
}