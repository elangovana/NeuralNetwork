using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class SigmoidActivation : IActivation
    {
        public double CalculateActivate(double x)
        {
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0/(1.0 + Math.Exp(-x));
        }

        public double CalculateDerivative(double x)
        {
            return (1 - x)*x;
        }
    }
}