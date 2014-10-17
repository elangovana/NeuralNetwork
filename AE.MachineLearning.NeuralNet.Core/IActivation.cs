namespace AE.MachineLearning.NeuralNet.Core
{
    public interface IActivation
    {
        double CalculateActivate(double x);

        double CalculateDerivative(double x);
    }
}