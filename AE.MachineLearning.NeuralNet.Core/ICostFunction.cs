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


        /// <summary>
        /// Derivative of cost function
        /// </summary>
        /// <param name="target">Expected Value</param>
        /// <param name="actual">Actual Value</param>
        /// <returns>Returns the derivate of cost function</returns>
        double DerivativeCostWrtOutput(double target, double actual);

      
    }
}