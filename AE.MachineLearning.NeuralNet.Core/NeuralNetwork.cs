using System;
using System.Runtime.Serialization;

namespace AE.MachineLearning.NeuralNet.Core
{
    [DataContract]
    public class NeuralNetwork
    {
        [DataMember] private readonly int _numberOfHiddenLayers;

        [DataMember] private readonly int _numberOfInputFeatures;

        [DataMember] private readonly int _numberOfOutputs;

        [DataMember] private readonly int[] _numberOfneuronsForHiddenLayers;
        private IActivation _activationOutput;

        [DataMember] private NetworkLayer[] _networkLayers;

        public NeuralNetwork()
        {
        }

        /// <summary>
        ///     Constructs a new neural network.
        /// </summary>
        /// <param name="numberOfInputFeatureFeatures">The number of input features</param>
        /// <param name="numberOfOutputs">The number of outputs</param>
        /// <param name="numberOfHiddenLayers">The number of hidden layers</param>
        /// <param name="numberOfneuronsForHiddenLayers">A array specifing the number of neurons for each hidden layer. The length of this array must match the number of hidden layers</param>
        /// <param name="activation">Activation function to use for the network</param>
        /// <param name="activationOutput">Optional Parameter, required only if a separate activation function is used by the output layer</param>
        public NeuralNetwork(int numberOfInputFeatureFeatures, int numberOfOutputs, int numberOfHiddenLayers,
                             int[] numberOfneuronsForHiddenLayers, IActivation activation,
                             IActivation activationOutput = null)
        {
            _numberOfInputFeatures = numberOfInputFeatureFeatures;
            _numberOfOutputs = numberOfOutputs;
            _numberOfHiddenLayers = numberOfHiddenLayers;
            _numberOfneuronsForHiddenLayers = numberOfneuronsForHiddenLayers;
            Activation = activation;
            ActivationOutput = activationOutput;
            if (numberOfneuronsForHiddenLayers.Length != numberOfHiddenLayers)
                throw new NeuralNetException(
                    string.Format(
                        "The length {0} of the numberOfneuronsForHiddenLayers array must be equal to the numberOfhiddenLayers specified {1} ",
                        numberOfneuronsForHiddenLayers.Length, numberOfHiddenLayers));
        }

        public IActivation Activation { get; private set; }


        public int NumberOfInputFeatures
        {
            get { return _numberOfInputFeatures; }
        }

        public int NumberOfOutputs
        {
            get { return _numberOfOutputs; }
        }

        public int NumberOfHiddenLayers
        {
            get { return _numberOfHiddenLayers; }
        }

        public int[] NumberOfneuronsForHiddenLayers
        {
            get { return _numberOfneuronsForHiddenLayers; }
        }

        public NetworkLayer OutputLayer
        {
            get { return NetworkLayers[NumberOfHiddenLayers + 1]; }
        }

        public NetworkLayer[] NetworkLayers
        {
            get { return _networkLayers ?? (_networkLayers = ConstructNetwork()); }
        }

        public IActivation ActivationOutput
        {
            get { return _activationOutput ?? Activation; }
            private set { _activationOutput = value; }
        }

        public double[] GetOutput()
        {
            var output = new double[NumberOfOutputs];

            for (int nu = 0; nu < OutputLayer.Neurons.Length; nu++)
            {
                Neuron neuron = NetworkLayers[NumberOfHiddenLayers + 1].Neurons[nu];
                output[nu] = neuron.Output;
            }

            return output;
        }


        public void ComputeOutput(double[] inputFeatures)
        {
            //Initially Previous layer out is the input itself
            var outPutOfPreviousLayer = new double[inputFeatures.Length];
            NetworkLayer inputLayer = NetworkLayers[0];
            for (int index = 0; index < inputFeatures.Length; index++)
            {
                double inputFeature = inputFeatures[index];
                inputLayer.Neurons[index].CalculateOutput(new[] {inputFeature});
                outPutOfPreviousLayer[index] = inputLayer.Neurons[index].Output;
            }

            //Rest of the layers
            for (int nw = 1; nw < NetworkLayers.Length; nw++)
            {
                var outputOfCurrentLayer = new double[NetworkLayers[nw].Neurons.Length];

                for (int nu = 0; nu < NetworkLayers[nw].Neurons.Length; nu++)
                {
                    Neuron neuron = NetworkLayers[nw].Neurons[nu];
                    neuron.CalculateOutput(outPutOfPreviousLayer);
                    outputOfCurrentLayer[nu] = neuron.Output;
                }
                outPutOfPreviousLayer = outputOfCurrentLayer;
            }
        }

        /// <summary>
        ///     Sets the weights for a given layer
        /// </summary>
        /// <param name="layerIndex">The layer index. The input layer is index 0.</param>
        /// <param name="weights">A two dimensional array of weights. The first dimension is the number of neurons, and the second dimension is the number of weights per neuron.</param>
        /// <param name="biases">The bias array. The length of this array is equal to the number of neurons in this layer</param>
        public void SetWeightsForLayer(int layerIndex, double[][] weights, double[] biases)
        {
            NetworkLayers[layerIndex].SetWeights(weights, biases);
        }


        public void InitNetworkWithRandomWeights(int? seed = default (int?))
        {
            Random random = seed == null ? new Random() : new Random(seed.Value);

            for (int nw = 1; nw < NetworkLayers.Length; nw++)
            {
                NetworkLayer layer = NetworkLayers[nw];
                var layerWeights = new double[layer.Neurons.Length][];
                var biases = new double[layer.Neurons.Length];
                for (int nu = 0; nu < layerWeights.Length; nu++)
                {
                    layerWeights[nu] = new double[NetworkLayers[nw - 1].NumOfNeurons];
                    for (int w = 0; w < layerWeights[nu].Length; w++)
                    {
                        layerWeights[nu][w] = random.NextDouble();
                    }
                    biases[nu] = random.NextDouble();
                }
                SetWeightsForLayer(nw, layerWeights, biases);
            }
        }

        private NetworkLayer[] ConstructNetwork()
        {
            int totalNumberOfLayers = _numberOfHiddenLayers + 2;
            var layers = new NetworkLayer[totalNumberOfLayers];

            //First layer -> tied to number of features
            layers[0] = new NetworkLayer(_numberOfInputFeatures, 1, new InputActivation(), true);

            //Hidden Layers
            for (int i = 1; i <= _numberOfHiddenLayers; i++)
            {
                layers[i] = new NetworkLayer(_numberOfneuronsForHiddenLayers[i - 1],
                                             layers[i - 1].NumOfNeurons,
                                             Activation);
            }

            //Final output layer -> tied to number of outputs
            layers[totalNumberOfLayers - 1] = new NetworkLayer(_numberOfOutputs,
                                                               layers[totalNumberOfLayers - 2].NumOfNeurons,
                                                               ActivationOutput);
            return layers;
        }

        public void PersistNetwork(string fileName)
        {
            PersistanceHelper.Serlialse(this, fileName);
        }

        public NeuralNetwork LoadNetwork(string fileName, IActivation activation, IActivation outputActivation = null)
        {
            Activation = activation;
            ActivationOutput = ActivationOutput;

            var networK = PersistanceHelper.Deseralise<NeuralNetwork>(fileName);

            networK.NetworkLayers[0].Activation = new InputActivation();

            for (var i = 1; i < networK.NetworkLayers.Length -1 ; i++)
            {
                networK.NetworkLayers[i].Activation = Activation;
                
            }

            networK.OutputLayer.Activation = ActivationOutput;


            return networK;
        }
    }
}