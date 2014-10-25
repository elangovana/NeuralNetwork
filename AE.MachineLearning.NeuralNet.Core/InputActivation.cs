namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Dummy Activation function for input layer. Applies no transformation!!
    /// </summary>
    internal class InputActivation : IActivation
    {
        public double CalculateActivate(double x)
        {
            return x;
        }

        public double CalculateDerivative(double x)
        {
            return 1.0;
        }
    }
}