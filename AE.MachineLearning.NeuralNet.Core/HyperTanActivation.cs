using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class HyperTanActivation : IActivation
    {
        public double CalculateActivate(double x)
        {
            if (x < -10.0) return -1.0;
            if (x > 10.0) return 1.0;
            return Math.Tanh(x);
        }

        public double CalculateDerivative(double x)
        {
           return (1 - x)*(1+x);
        }
    }
}