namespace AE.MachineLearning.NeuralNet.Core
{
    public interface INetworkFactory
    {
        IActivation Activation { get; set; }
        int NumberOfInputFeatures { get; set; }
        int NumberOfOutputs { get; set; }
        int NumberOfHiddenLayers { get; set; }
        int[] NumberOfneuronsForHiddenLayers { get; set; }
        NetworkLayer OutputLayer { get; }
        IActivation ActivationOutput { get; set; }

        AbstractNetwork CreateNetwork();
    }
}