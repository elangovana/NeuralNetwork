namespace AE.MachineLearning.NeuralNet.Core
{
    public interface IErrorCalculator
    {
        /// <summary>
        /// Caculates error rate by comparing the target or expected values with the actual outputs
        /// </summary>
        /// <param name="targetOutputs">Expected outputs</param>
        /// <param name="actutalOutputs">Actual outputs</param>
        /// <returns>An error rate</returns>
        double CalculateError(double[][] targetOutputs, double[][] actutalOutputs);
    }
}