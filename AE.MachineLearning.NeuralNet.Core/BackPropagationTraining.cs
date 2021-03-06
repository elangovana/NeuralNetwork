﻿using System;
using System.IO;
using System.Linq;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class BackPropagationTraining : ITrainingAlgoritihm
    {
        private readonly IGradientCalculator _gradientCalculator;
        private IErrorCalculator _errorCalculator;

        private int _flushCounter;
        private double[][] _gradients;
        private double _maxError = .05;
        private int _maxIteration = 10000;
        private double[][] _previousDeltaBias;
        private double[][][] _previousDeltaWeight;
        private IShuffler _shouffler;
 

        public BackPropagationTraining(IGradientCalculator gradientCalculator, string outDir = null)
        {
            OutDir = outDir;
            _gradientCalculator = gradientCalculator;
        }

        public BackPropagationTraining(AbstractNetwork network, IGradientCalculator gradientCalculator,
                                       string outDir = null)
        {
            Network = network;
            _gradientCalculator = gradientCalculator;
            OutDir = outDir;
        }


       
        public string OutDir { get; set; }

        public double LearningRate { get; set; }

        public double Momentum { get; set; }

        public double MaxError
        {
            get { return _maxError; }
            set { _maxError = value; }
        }


        public int MaxIteration
        {
            get { return _maxIteration; }
            set { _maxIteration = value; }
        }

        public int LogLevel { get; set; }

        public IShuffler Shouffler
        {
            get { return _shouffler ?? (_shouffler = new Shuffler()); }
            set { _shouffler = value; }
        }

        public IErrorCalculator ErrorCalculator
        {
            get { return _errorCalculator ?? (_errorCalculator = new ClassificationErrorCalculator()); }
            set { _errorCalculator = value; }
        }

        public StreamWriter LogWriter { get; set; }

        public AbstractNetwork Network { get; set; }


        /// <summary>
        ///     Trains the network
        /// </summary>
        /// <param name="inputs">The first dimenension is the dataset, the second dimension must be equal  number of input features </param>
        /// <param name="targetOutputs"></param>
        public void Train(double[][] inputs, double[][] targetOutputs)
        {
            if (inputs.Any(x => x.Length != Network.NumberOfInputFeatures))
                throw new NeuralNetException(
                    string.Format(
                        "Each item in the input array data must contain number of features specified in the constructor {0}",
                        Network.NumberOfInputFeatures));

            if (targetOutputs.Any(x => x.Length != Network.NumberOfOutputs))
                throw new NeuralNetException(
                    string.Format(
                        "Each item in the output array data must contain number of output specified in the constructor {0}",
                        Network.NumberOfOutputs));

            if (inputs.Length != targetOutputs.Length)
                throw new NeuralNetException(
                    string.Format(
                        "The length {0} of the input vector does not match the length {1} of the target output vector",
                        inputs.Length, targetOutputs.Length));

            _gradients = new double[Network.NetworkLayers.Length][];

            InitPreviousDeltas();

            WriteLog(
                string.Format(
                    "Back Propgation settings learningRate {0}, Momentum {1}, MaxError {2}, MaxIterations{3} ",
                    LearningRate, Momentum, MaxError, MaxIteration));
            int iter = 0;
            double error;
            double minError = 1000.0;
            string bestNetworkFile = "";
            AbstractNetwork bestNetwork = null;
            double[][] shouffledInputs = inputs;
            double[][] shouffledTargetOutputs = targetOutputs;
            do
            {
                //Shuffle Inputs
                double[][] tI;
                double[][] to;
                Shouffler.Shuffle(shouffledInputs, shouffledTargetOutputs, out tI, out to);
                shouffledInputs = tI;
                shouffledTargetOutputs = to;

                //Run Back Prop Stoch Gradient
                RunStochasticGradientBackProp(shouffledInputs, shouffledTargetOutputs);

                //validate
                double[][] actualOutputs = Predict(shouffledInputs);

                //Calc Error
                error = ErrorCalculator.CalculateError(shouffledTargetOutputs, actualOutputs);
                iter++;
                var fileName = string.Format("Network{0}.xml", iter);
                Persist(fileName);
                if (error < minError)
                {
                    minError = error;
                    bestNetworkFile = fileName;
                    bestNetwork = Network.CloneNetwork(Network);
                }
                WriteLog(string.Format("Iteration {0},  % correct {1}", iter, (1.0 - error)*100.0),2);
           
                //loop till target error is achieved
            } while (error > MaxError && iter < MaxIteration);

            Network = bestNetwork;
            WriteLog(string.Format("BestNetwork {0}, with error {1}", bestNetworkFile, minError));
        }


        /// <summary>
        ///     Predicts output based on the training
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[][] Predict(double[][] inputs)
        {
            var output = new double[inputs.Length][];

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] input = inputs[i];

                Network.ComputeOutput(input);

                output[i] = Network.GetOutput();
            }

            return output;
        }

        private void Persist(string fileName)
        {
            if (!string.IsNullOrEmpty(OutDir))
            {
                Network.PersistNetwork(Path.Combine(OutDir, fileName));
            }
        }

        private void RunStochasticGradientBackProp(double[][] inputs, double[][] targetOutputs)
        {
            for (int index = 0; index < inputs.Length; index++)
            {
                //Compute output
                Network.ComputeOutput(inputs[index]);

                //Cal errror to keep going :-)
                // error += CalcError(targetOutputs[index], Network.GetOutput());

                // error = error/(iter + 1);
                //Compute gradient for output layer
                int nw = Network.NetworkLayers.Length - 1;
                _gradients[nw] = ComputeGradientOutput(Network.NetworkLayers[nw], targetOutputs[index]);

                //Compute gradient for rest of the layers
                for (nw = nw - 1; nw > 0; nw--)
                {
                    _gradients[nw] = ComputeGradientHidden(Network.NetworkLayers[nw + 1], _gradients[nw + 1],
                                                           Network.NetworkLayers[nw]);
                }

                UpdateWeights(LearningRate, Momentum);
            }
        }

        private void WriteLog(string message, int logLevel)
        {
            if (LogLevel <= logLevel) WriteLog(message);
        }

        private void InitPreviousDeltas()
        {
            _previousDeltaWeight = new double[Network.NetworkLayers.Length][][];
            _previousDeltaBias = new double[Network.NetworkLayers.Length][];
            for (int nw = 0; nw < Network.NetworkLayers.Length; nw++)
            {
                _previousDeltaWeight[nw] = new double[Network.NetworkLayers[nw].Neurons.Length][];
                _previousDeltaBias[nw] = new double[Network.NetworkLayers[nw].Neurons.Length];

                for (int nu = 0; nu < Network.NetworkLayers[nw].Neurons.Length; nu++)
                {
                    int lengthOfWeights = Network.NetworkLayers[nw].Neurons[nu].Weights.Length;
                    _previousDeltaWeight[nw][nu] = new double[lengthOfWeights];
                }
            }
        }

        /// <summary>
        ///     Computes the gradient in the final output layer
        /// </summary>
        /// <param name="outputLayer">The output network layer</param>
        /// <param name="outputValues">The expected  or target values for each neuron in this layer for each input data.</param>
        /// <returns>the gradient of the output layer (dy/dz) * (de/dy)</returns>
        private double[] ComputeGradientOutput(NetworkLayer outputLayer, double[] outputValues)
        {
            var gradientToCompute = new double[outputLayer.NumOfNeurons];

            for (int n = 0; n < gradientToCompute.Length; ++n)
            {
                gradientToCompute[n] = _gradientCalculator.CalculateGradientOutputLayer(outputValues[n],
                                                                                        outputLayer.Neurons[n].Output);
            }

            return gradientToCompute;
        }


        /// <summary>
        ///     Computes the gradient for hidden layers
        /// </summary>
        /// <param name="nextLayer">The "next layer" wrt feedforward. For instance, the output layer would be the next next for the just previosu hidden layer</param>
        /// <param name="gradientsOfNextLayer">The gradients of the next layer , previously calculated in back propogation</param>
        /// <param name="layer">The layer for which the gradients need to be computed</param>
        /// <returns></returns>
        /// The gradient to compute contains the result of the calulation
        private double[] ComputeGradientHidden(NetworkLayer nextLayer, double[] gradientsOfNextLayer,
                                               NetworkLayer layer)
        {
            IActivation activationFunction = layer.Activation;
            var gradientToCompute = new double[layer.NumOfNeurons];
            for (int i = 0; i < gradientToCompute.Length; ++i)
            {
                //derivative of this neuron
                double derivative = activationFunction.CalculateDerivative(layer.Neurons[i].Output);

                //next layers total error contribution from this neuron
                double errorContribToNextLayer = 0;
                for (int j = 0; j < nextLayer.Neurons.Length; j++)
                {
                    errorContribToNextLayer += nextLayer.Neurons[j].Weights[i]*gradientsOfNextLayer[j];
                }

                //Add derivate to previous
                gradientToCompute[i] = derivative*errorContribToNextLayer;
            }

            return gradientToCompute;
        }

        private double UpdateWeights(double learningRate, double momentum)
        {
            double totalGradient = 0.0;
            for (int nw = 1; nw < Network.NetworkLayers.Length; nw++)
            {
                NetworkLayer layer = Network.NetworkLayers[nw];
                for (int nu = 0; nu < layer.Neurons.Length; nu++)
                {
                    Neuron neuron = layer.Neurons[nu];
                    totalGradient += Math.Abs(_gradients[nw][nu]);
                    double deltaBias = learningRate*_gradients[nw][nu];
                    neuron.Bias += deltaBias + momentum*_previousDeltaBias[nw][nu];
                    _previousDeltaBias[nw][nu] = deltaBias;

                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        // gradient of for this neuron * input for this neuron
                        double delta = learningRate*_gradients[nw][nu]*Network.NetworkLayers[nw - 1].Neurons[w].Output;
                        neuron.Weights[w] += delta + momentum*_previousDeltaWeight[nw][nu][w];
                        _previousDeltaWeight[nw][nu][w] = delta;
                    }
                }
            }

            return totalGradient;
        }


        private void WriteLog(string message)
        {
            if (LogWriter == null) return;

            LogWriter.WriteLine("{0} - {1}", DateTime.Now, message);
            if (_flushCounter%10 == 0)
            {
                LogWriter.Flush();
                _flushCounter = 0;
            }
            _flushCounter++;
        }
    }
}