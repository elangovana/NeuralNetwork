namespace AE.MachineLearning.NeuralNet.Core
{
    public class BackPropagationTraining
    {
        private readonly ICostFunction _costFunction;
        private readonly double[][] _gradients;
        private readonly NeuralNetwork _network;
        private const double Error = .01;

        public BackPropagationTraining(NeuralNetwork network, ICostFunction costFunction)
        {
            _network = network;
            _costFunction = costFunction;
            _gradients = new double[network.NetworkLayers.Length - 1][];
        }

        /// <summary>
        ///     Trains the network
        /// </summary>
        /// <param name="inputs">The first dimenension is the dataset, the second dimension must be equal  number of input features </param>
        /// <param name="targetOutputs"></param>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        public void Train(double[][] inputs, double[][] targetOutputs, double learningRate, double momentum)
        {
            if (inputs.GetLength(1) != _network.NumberOfInputFeatures)
                throw new NeuralNetException(
                    string.Format(
                        "The length {0} second dimension of the input array is must be number of features specified in the constructor {1}",
                        inputs.GetLength(1), _network.NumberOfInputFeatures));

            //TODO: Set init Weights!! Either Randomise or accept weights

            for (int index = 0; index < inputs.Length; index++)
            {
                //Compute output
                _network.ComputeOutput(inputs[index]);

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
                UpdateWeights(learningRate, momentum);
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

            IActivation activationFunction = outputLayer.Activation;
            for (int n = 0; n < gradientToCompute.Length; ++n)
            {
                double derivativeActivation = activationFunction.CalculateDerivative(outputValues[n]);
                gradientToCompute[n] = derivativeActivation*
                                       _costFunction.DerivativeCostWrtOutput(outputValues[n],
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
                    errorContribToNextLayer = nextLayer.Neurons[j].Weights[i]*gradientsOfNextLayer[j];
                }

                //Add derivate to previous
                gradientToCompute[i] = derivative*errorContribToNextLayer;
            }

            return gradientToCompute;
        }

        public void UpdateWeights(double learningRate, double momentum)
        {
            for (int nw = 1; nw < _network.NetworkLayers.Length; nw++)
            {
                var layer = _network.NetworkLayers[nw];
                for (int nu = 0; nu < layer.Neurons.Length; nu++)
                {
                    Neuron neuron = layer.Neurons[nu];
                    neuron.Bias += learningRate*_gradients[nw][nu];
                        //todo: add momemtum to bias
                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        // gradient of for this neuron * input for this neuron
                        double delta = learningRate*_gradients[nw][nu]*_network.NetworkLayers[nw - 1].Neurons[w].Output;
                        neuron.Weights[w] += delta;
                        //TODO add momentum
                       // neuron.Weights[w] = newWeight + momentum * neuron.Weights[w];
                    }
                }
            }
           
        }
    }
}