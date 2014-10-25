using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    /// <summary>
    ///     Calculates Fitness by percentage correct
    /// </summary>
    public class ClassficationFitnessCalculator : IFitnessCalculator
    {
        /// <summary>
        /// Returns % correct
        /// </summary>
        /// <param name="targetOutputs">Expected outputs</param>
        /// <param name="actualOutputs">Actual outputs</param>
        /// <returns></returns>
        public double Calculator(double[][] targetOutputs, double[][] actualOutputs)
        {
            double error = new ClassificationErrorCalculator().CalculateError(targetOutputs, actualOutputs);

            return (1.0 - error)*100;
        }
    }
}