using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Activation function TanH
    /// </summary>
    public class HyperTanActivation : IActivation
    {
        /// <summary>
        /// Cacluate activation value with Math.tanh(x)
        /// </summary>
        /// <param name="x">Input  value</param>
        /// <returns>Activation values at x, using Math.tanh(x)</returns>
        public double CalculateActivate(double x)
        {
            if (x < -10.0) return -1.0;
            if (x > 10.0) return 1.0;
            return Math.Tanh(x);
        }

        /// <summary>
        /// Computes the Derivate of Tanh(x)
        /// </summary>
        /// <param name="x">Value to compute the derivate for</param>
        /// <returns>the derivate of tanh at the input value</returns>
        public double CalculateDerivative(double x)
        {
           return (1 - x)*(1+x);
        }
    }
}