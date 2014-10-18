namespace AE.MachineLearning.NeuralNet.Core
{
    public interface ICostFunction
    {
        /// <summary>
        ///     Cost (Error per training data)
        /// </summary>
        /// <param name="target">expected value of output</param>
        /// <param name="actual">actual Value or output</param>
        /// <returns>Cost</returns>
        double Cost(double target, double actual);

        double Cost(double[] target, double[] actual);

        double DerivativeCostWrtOutput(double target, double actual);

        double DerivativeCostWrtOutput(double[] target, double[] actual);
    }
}