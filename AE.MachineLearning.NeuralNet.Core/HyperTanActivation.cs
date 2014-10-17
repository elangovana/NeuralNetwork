using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class HyperTanActivation : IActivate
    {
        public double Activate(double x)
        {
            if (x < -10.0) return -1.0;
            if (x > 10.0) return 1.0;
            return Math.Tanh(x);
        }
    }
}