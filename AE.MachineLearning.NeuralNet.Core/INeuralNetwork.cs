namespace AE.MachineLearning.NeuralNet.Core
{
    internal interface INeuralNetwork
    {

        void UpdateWeights(double[] targetValues, double learningRate, double momentum);
        void SetWeights(double[] weights);
        double[] GetWeights();
        double[] ComputeOutputs(double[] xValues);
    }
}