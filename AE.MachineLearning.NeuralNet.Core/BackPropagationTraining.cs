using System;
using System.IO;
using System.Linq;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class BackPropagationTraining
    {
        private const double Error = .01;
        private readonly IGradientCalculator _gradientCalculator;

        private readonly double[][] _gradients;
        private readonly NeuralNetwork _network;
        private int _flushCounter;
        private double[][] _previousDeltaBias;
        private double[][][] _previousDeltaWeight;

        public BackPropagationTraining(NeuralNetwork network, IGradientCalculator gradientCalculator)
        {
            _network = network;
            _gradientCalculator = gradientCalculator;

            _gradients = new double[network.NetworkLayers.Length][];


            InitPreviousDeltas();
        }

        public StreamWriter LogWriter { get; set; }

        private void InitPreviousDeltas()
        {
            _previousDeltaWeight = new double[_network.NetworkLayers.Length][][];
            _previousDeltaBias = new double[_network.NetworkLayers.Length][];
            for (int nw = 0; nw < _network.NetworkLayers.Length; nw++)
            {
                _previousDeltaWeight[nw] = new double[_network.NetworkLayers[nw].Neurons.Length][];
                _previousDeltaBias[nw] = new double[_network.NetworkLayers[nw].Neurons.Length];

                for (int nu = 0; nu < _network.NetworkLayers[nw].Neurons.Length; nu++)
                {
                    int lengthOfWeights = _network.NetworkLayers[nw].Neurons[nu].Weights.Length;
                    _previousDeltaWeight[nw][nu] = new double[lengthOfWeights];
                }
            }
        }

        /// <summary>
        ///     Trains the network
        /// </summary>
        /// <param name="inputs">The first dimenension is the dataset, the second dimension must be equal  number of input features </param>
        /// <param name="targetOutputs"></param>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        /// <param name="maxError">Max Error Value. Stops when the error rate reaches this value</param>
        /// <param name="maxIteration">To prevent long running times, stops at this iteration</param>
        public void Train(double[][] inputs, double[][] targetOutputs, double learningRate, double momentum,
                          double maxError = .05, int maxIteration = 10000)
        {
            if (inputs.Any(x => x.Length != _network.NumberOfInputFeatures))
                throw new NeuralNetException(
                    string.Format(
                        "Each item in the input array data must contain number of features specified in the constructor {0}",
                        _network.NumberOfInputFeatures));

            if (targetOutputs.Any(x => x.Length != _network.NumberOfOutputs))
                throw new NeuralNetException(
                    string.Format(
                        "Each item in the output array data must contain number of output specified in the constructor {0}",
                        _network.NumberOfOutputs));

            if (inputs.Length != targetOutputs.Length)
                throw new NeuralNetException(
                    string.Format(
                        "The length {0} of the input vector does not match the length {1} of the target output vector",
                        inputs.Length, targetOutputs.Length));


            for (int index = 0; index < inputs.Length; index++)
            {
                double error = 0.0;
                int iter = 0;
                double totalGradientChange = 0.0;
                do
                {
                    //Compute output
                    _network.ComputeOutput(inputs[index]);

                    //Cal errror to keep going :-)
                    error += CalcError(targetOutputs[index], _network.GetOutput());

                    error = error/(iter + 1);
                    //Compute gradient for output layer
                    int nw = _network.NetworkLayers.Length - 1;
                    _gradients[nw] = ComputeGradientOutput(_network.NetworkLayers[nw], targetOutputs[index]);

                    //Compute gradient for rest of the layers
                    for (nw = nw - 1; nw > 0; nw--)
                    {
                        _gradients[nw] = ComputeGradientHidden(_network.NetworkLayers[nw + 1], _gradients[nw + 1],
                                                               _network.NetworkLayers[nw]);
                    }


                    //Update Weights
                    totalGradientChange += UpdateWeights(learningRate, momentum);
                    totalGradientChange = totalGradientChange/(iter + 1);

                    iter++;
                } while (totalGradientChange > maxError && iter < maxIteration);
                WriteLog(string.Format("Input Index {2}, Iteration {0}, Average Error {1}, Total Gradient Change {3}", iter,
                                       error, index, totalGradientChange.ToString("F5")));
            }
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

                _network.ComputeOutput(input);

                output[i] = _network.GetOutput();
            }

            return output;
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
            for (int nw = 1; nw < _network.NetworkLayers.Length; nw++)
            {
                NetworkLayer layer = _network.NetworkLayers[nw];
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
                        double delta = learningRate*_gradients[nw][nu]*_network.NetworkLayers[nw - 1].Neurons[w].Output;
                        neuron.Weights[w] += delta + momentum*_previousDeltaWeight[nw][nu][w];
                        _previousDeltaWeight[nw][nu][w] = delta;
                    }
                }
            }

            return totalGradient;
        }


        private static double CalcError(double[] target, double[] output)
        {
            return (target.Select((t, i) => Math.Abs(t - output[i])).Sum());
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