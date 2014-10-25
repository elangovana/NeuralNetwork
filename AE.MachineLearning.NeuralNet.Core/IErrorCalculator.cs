namespace AE.MachineLearning.NeuralNet.Core
{
    public interface IErrorCalculator
    {
        double CalculateError(double[][] targetOutputs, double[][] actutalOutputs);
    }
}