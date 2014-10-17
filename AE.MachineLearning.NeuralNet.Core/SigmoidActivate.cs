using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    internal class SigmoidActivate : IActivate
    {
        public double Activate(double x)
        {
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0/(1.0 + Math.Exp(-x));
        }
    }
}