﻿using System.Linq;
using System.Runtime.Serialization;

namespace AE.MachineLearning.NeuralNet.Core
{
    [DataContract]
    public class NetworkLayer
    {
        [DataMember] private readonly bool _isInputLayer;

        [DataMember] private readonly Neuron[] _neurons;

        [DataMember] private readonly int _numOfNeurons;

        [DataMember] private readonly int _numberOfInputsPerNeuron;

        private IActivation _activation;

        public NetworkLayer()
        {
        }

        /// <summary>
        ///     Initialises a networklayer
        /// </summary>
        /// <param name="numOfNeurons">the number of neurons in this layer</param>
        /// <param name="numberOfInputsPerNeuron">This is the number of neurons in the previous layer and is thus the number of inputs to each neuron in this layer</param>
        /// <param name="activation">Activation Function</param>
        /// <param name="isInputLayer"></param>
        public NetworkLayer(int numOfNeurons, int numberOfInputsPerNeuron, IActivation activation,
                            bool isInputLayer = false)
        {
            _numOfNeurons = numOfNeurons;
            _numberOfInputsPerNeuron = numberOfInputsPerNeuron;
            _activation = activation;
            _isInputLayer = isInputLayer;
            _neurons = new Neuron[numOfNeurons];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(Activation);
            }

            if (!isInputLayer) return;
            var inputWeights = new double[numOfNeurons][];
            for (int index = 0; index < inputWeights.Length; index++)
            {
                inputWeights[index] = new[] {1.0};
            }
            SetWeights(inputWeights, new double[numOfNeurons]);
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
            set
            {
                _activation = value;
                UpdateNeuronActivations();
            }
        }

        public bool IsInputLayer
        {
            get { return _isInputLayer; }
        }

        private void UpdateNeuronActivations()
        {
            foreach (Neuron neuron in Neurons)
            {
                neuron.Activation = Activation;
            }
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