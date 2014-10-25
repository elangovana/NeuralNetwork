using System.Runtime.Serialization;

namespace AE.MachineLearning.NeuralNet.Core
{
    [DataContract]
    public abstract class AbstractNetwork
    {
        public abstract IActivation Activation { get;  }
        public abstract int NumberOfInputFeatures { get;  }
        public abstract int NumberOfOutputs { get; }
        public abstract int NumberOfHiddenLayers { get;  }
        public abstract int[] NumberOfneuronsForHiddenLayers { get;  }
        public abstract NetworkLayer OutputLayer { get; }
        public abstract IActivation ActivationOutput { get; }
        public abstract double[] GetOutput();
        public abstract void ComputeOutput(double[] inputFeatures);

        public abstract NetworkLayer[] NetworkLayers { get; }

        public abstract void AddNeuron(int layerIndex);

        public abstract void DeleteNeuron(int layerIndex);

        public abstract AbstractNetwork CloneNetwork(AbstractNetwork networkToClone);


        /// <summary>
        ///     Sets the weights for a given layer
        /// </summary>
        /// <param name="layerIndex">The layer index. The input layer is index 0.</param>
        /// <param name="weights">A two dimensional array of weights. The first dimension is the number of neurons, and the second dimension is the number of weights per neuron.</param>
        /// <param name="biases">The bias array. The length of this array is equal to the number of neurons in this layer</param>
        public abstract void SetWeightsForLayer(int layerIndex, double[][] weights, double[] biases);

        public abstract void InitNetworkWithRandomWeights(int? seed = default (int?));
        public abstract void PersistNetwork(string fileName);
        public abstract AbstractNetwork LoadNetwork(string fileName, IActivation activation, IActivation outputActivation = null);
    }
}