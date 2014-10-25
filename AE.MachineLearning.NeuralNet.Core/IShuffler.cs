namespace AE.MachineLearning.NeuralNet.Core
{
    public interface IShuffler
    {
        void Shouffle(double[][] inputs, double[][] outputs, out double[][] randomisedInputs,
                      out double[][] randomisedOutputs);
    }
}