namespace AE.MachineLearning.NeuralNet.Core
{
    public class NeuralNetwork
    {
        private readonly IActivation _activation;
        private readonly IActivation _activationOutput;
        private readonly NetworkLayer[] _networkLayers;
        private readonly int _numberOfHiddenLayers;
        private readonly int _numberOfInputFeatures;
        private readonly int _numberOfOutputs;
        private readonly int[] _numberOfneuronsForHiddenLayers;

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
                             int[] numberOfneuronsForHiddenLayers, IActivation activation, IActivation activationOutput = null)
        {
            _numberOfInputFeatures = numberOfInputFeatureFeatures;
            _numberOfOutputs = numberOfOutputs;
            _numberOfHiddenLayers = numberOfHiddenLayers;
            _numberOfneuronsForHiddenLayers = numberOfneuronsForHiddenLayers;
            _activation = activation;
            _activationOutput = activationOutput ?? _activation;
            if (numberOfneuronsForHiddenLayers.Length != numberOfHiddenLayers)
                throw new NeuralNetException(
                    string.Format(
                        "The length {0} of the numberOfneuronsForHiddenLayers array must be equal to the numberOfhiddenLayers specified {1} ",
                        numberOfneuronsForHiddenLayers.Length, numberOfHiddenLayers));

            _networkLayers = ConstructNetwork();
        }


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

        public NetworkLayer[] NetworkLayers
        {
            get { return _networkLayers; }
        }

        public IActivation ActivationOutput
        {
            get { return _activationOutput; }
        }



        public void ComputeOutput(double[] inputFeatures)
        {
//Initially Previous layer out is the input itself
            double[] outPutOfPreviousLayer = inputFeatures;

            for (int i = 0; i < NetworkLayers.Length; i++)
            {
                var outputOfCurrentLayer = new double[NetworkLayers[i].Neurons.Length];

                foreach (Neuron neuron in NetworkLayers[i].Neurons)
                {
                    neuron.CalculateOutput(outPutOfPreviousLayer);
                    outputOfCurrentLayer[i] = neuron.Output;
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

        private NetworkLayer[] ConstructNetwork()
        {
            int totalNumberOfLayers = _numberOfHiddenLayers + 2;
            var layers = new NetworkLayer[totalNumberOfLayers];

            //First layer -> tied to number of features
            layers[0] = new NetworkLayer(_numberOfInputFeatures, 1, new InputActivation());

            //Hidden Layers
            for (int i = 1; i <= _numberOfHiddenLayers; i++)
            {
                layers[i] = new NetworkLayer(_numberOfneuronsForHiddenLayers[i - 1],
                                             layers[i - 1].NumOfNeurons,
                                             _activation);
            }

            //Final output layer -> tied to number of outputs
            layers[totalNumberOfLayers - 1] = new NetworkLayer(_numberOfOutputs,
                                                               layers[totalNumberOfLayers - 2].NumOfNeurons,
                                                               ActivationOutput);
            return layers;
        }
    }
}