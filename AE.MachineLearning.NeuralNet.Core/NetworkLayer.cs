using System.Linq;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class NetworkLayer
    {
        private readonly IActivation _activation;

        private readonly Neuron[] _neurons;
        private readonly int _numOfNeurons;
        private readonly int _numberOfInputsPerNeuron;

        /// <summary>
        ///     Initialises a networklayer
        /// </summary>
        /// <param name="numOfNeurons">the number of neurons in this layer</param>
        /// <param name="numberOfInputsPerNeuron">This is the number of neurons in the previous layer and is thus the number of inputs to each neuron in this layer</param>
        /// <param name="activation">Activation Function</param>
        public NetworkLayer(int numOfNeurons, int numberOfInputsPerNeuron, IActivation activation)
        {
            _numOfNeurons = numOfNeurons;
            _numberOfInputsPerNeuron = numberOfInputsPerNeuron;
            _activation = activation;
            _neurons = new Neuron[numOfNeurons];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(_activation);
            }
        }

        public Neuron[] Neurons
        {
            get { return _neurons; }
        }

        public int NumOfNeurons
        {
            get { return _numOfNeurons; }
        }

        public int NumberOfInputsPerNeuron
        {
            get { return _numberOfInputsPerNeuron; }
        }

        public IActivation Activation
        {
            get { return _activation; }
        }


        /// <summary>
        ///     Sets the weights and bias for each neuron
        /// </summary>
        /// <param name="weights">A two dimenional array of weights. The first dimension must be equal to the number of neurons in this layer. The Second dimsenion is equal to the number of inputs to each neuron or the number the neurons in the previous layer</param>
        /// <param name="bias">A one dimesional array of biases, one for each neuron in this layer.</param>
        public void SetWeights(double[][] weights, double[] bias)
        {
            ValidateVectors(weights, bias);

            for (int i = 0; i < NumOfNeurons; i++)
            {
                Neurons[i].Weights = weights[i];
                Neurons[i].Bias = bias[i];
            }
        }

        private void ValidateVectors(double[][] weights, double[] bias)
        {
//Validate lengths and sizes first!!
            if (bias.Length != NumOfNeurons)
                throw new NeuralNetException(
                    string.Format(
                        "The  length of the input bias vector  {0} must be  equal to the number of neurons {1} in this layer",
                        bias.Length, NumOfNeurons));


            if (weights.Length != NumOfNeurons)
                throw new NeuralNetException(
                    string.Format(
                        "The length {0}  first dimension of the input weights vector  must be  equal to the number of neurons  {1} in this layer",
                        weights.GetLength(0), NumOfNeurons));

            if (weights.Any(x => x.Length != NumberOfInputsPerNeuron))
                throw new NeuralNetException(
                    string.Format(
                        "The length  the second dimension of the input weights vector  must be  equal to the number of neurons  {0} in the previous layer",
                        NumberOfInputsPerNeuron));
        }
    }
}