namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Describes the activation function
    /// </summary>
    public interface IActivation
    {
        /// <summary>
        /// Calculates the actvation function for a given input.
        /// </summary>
        /// <param name="x">Input x at which the activation functions needs to be computed</param>
        /// <returns>The value of activation function at input x</returns>
        double CalculateActivate(double x);

        /// <summary>
        /// Computes the derivate of the activation function at input value x
        /// </summary>
        /// <param name="x">The input value to compute the derivatve at</param>
        /// <returns>The derivate of the actiation function x</returns>
        double CalculateDerivative(double x);
    }
}