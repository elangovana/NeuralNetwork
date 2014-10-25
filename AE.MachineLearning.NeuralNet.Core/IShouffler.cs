namespace AE.MachineLearning.NeuralNet.Core
{
    public interface IShouffler
    {
        void Shouffle(double[][] inputs, double[][] outputs, out double[][] randomisedInputs,
                      out double[][] randomisedOutputs);
    }
}