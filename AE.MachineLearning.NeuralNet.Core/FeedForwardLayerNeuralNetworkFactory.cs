﻿namespace AE.MachineLearning.NeuralNet.Core
{
    public class FeedForwardLayerNeuralNetworkFactory : INetworkFactory
    {
        public IActivation Activation { get; set; }
        public int NumberOfInputFeatures { get; set; }
        public int NumberOfOutputs { get; set; }
        public int NumberOfHiddenLayers { get; set; }
        public int[] NumberOfneuronsForHiddenLayers { get; set; }
        public NetworkLayer OutputLayer { get; set; }
        public IActivation ActivationOutput { get; set; }

        public AbstractNetwork CreateNetwork()
        {
            return new NeuralNetwork(NumberOfInputFeatures, NumberOfOutputs, NumberOfHiddenLayers,
                                     NumberOfneuronsForHiddenLayers, Activation, ActivationOutput);
        }
    }
}