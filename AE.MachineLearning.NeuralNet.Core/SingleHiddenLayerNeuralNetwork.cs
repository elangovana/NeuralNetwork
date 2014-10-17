using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// A Neural network implementation with just one hidden layer.
    /// </summary>
    public class SingleHiddenLayerNeuralNetwork : INeuralNetwork
    {
        private readonly IActivate _activation;
        private readonly double[] _hGrads; // hidden gradients for back-propagation
        private readonly double[] _hoBiases;
        private readonly double[][] _ihWeights; // input-to-hidden
        private readonly double[] _inputs;
        private readonly int _numHidden;
        private readonly int _numInput;
        private readonly int _numOutput;
        private readonly double[] _hoPrevBiasesDelta;
        private readonly double[][] _hoPrevWeightsDelta;
        private readonly double[] _hoSums;
        private readonly double[][] _hoWeights; // hidden-to-output

        private readonly double[] _ihBiases;
        private readonly double[] _ihOutputs;

        private readonly double[] _ihPrevBiasesDelta;
        private readonly double[][] _ihPrevWeightsDelta; // for momentum with back-propagation
        private readonly double[] _ihSums;
        private readonly double[] _oGrads; // output gradients for back-propagation
        private readonly double[] _outputs;

        /// <summary>
        /// Constructs a single hidden layer neural network given the  number of input, output and hidden neurons
        /// </summary>
        /// <param name="numInput">Number of inputs</param>
        /// <param name="numHidden">Number of neurons in the hidden layer</param>
        /// <param name="numOutput">number of outputs</param>
        /// <param name="activation">Activation function to use </param>
        public SingleHiddenLayerNeuralNetwork(int numInput, int numHidden, int numOutput, IActivate activation)
        {
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;
            _activation = activation;

            _inputs = new double[numInput];
            _ihWeights = MatrixHelper.CreateMatrix(numInput, numHidden);
            _ihSums = new double[numHidden];
            _ihBiases = new double[numHidden];
            _ihOutputs = new double[numHidden];
            _hoWeights = MatrixHelper.CreateMatrix(numHidden, numOutput);
            _hoSums = new double[numOutput];
            _hoBiases = new double[numOutput];
            _outputs = new double[numOutput];

            _oGrads = new double[numOutput];
            _hGrads = new double[numHidden];

            _ihPrevWeightsDelta = MatrixHelper.CreateMatrix(numInput, numHidden);
            _ihPrevBiasesDelta = new double[numHidden];
            _hoPrevWeightsDelta = MatrixHelper.CreateMatrix(numHidden, numOutput);
            _hoPrevBiasesDelta = new double[numOutput];
        }

        public void UpdateWeights(double[] tValues, double learningRate, double momentum)
            // update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
        {
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and matrices have values (other than 0.0)
            if (tValues.Length != _numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");


            ComputeGradientOutput(tValues);

            ComputeHiddenGradients();

            // 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order
            for (int i = 0; i < _ihWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < _ihWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = learningRate*_hGrads[j]*_inputs[i]; // compute the new delta
                    _ihWeights[i][j] += delta; // update
                    _ihWeights[i][j] += momentum*_ihPrevWeightsDelta[i][j];
                    // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                }
            }

            // 3b. update input to hidden biases
            for (int i = 0; i < _ihBiases.Length; ++i)
            {
                double delta = learningRate*_hGrads[i]*1.0;
                    // the 1.0 is the constant input for any bias; could leave out
                _ihBiases[i] += delta;
                _ihBiases[i] += momentum*_ihPrevBiasesDelta[i];
            }

            // 4. update hidden to output weights
            for (int i = 0; i < _hoWeights.Length; ++i) // 0..3 (4)
            {
                for (int j = 0; j < _hoWeights[0].Length; ++j) // 0..1 (2)
                {
                    double delta = learningRate*_oGrads[j]*_ihOutputs[i]; // see above: ihOutputs are inputs to next layer
                    _hoWeights[i][j] += delta;
                    _hoWeights[i][j] += momentum*_hoPrevWeightsDelta[i][j];
                    _hoPrevWeightsDelta[i][j] = delta;
                }
            }

            // 4b. update hidden to output biases
            for (int i = 0; i < _hoBiases.Length; ++i)
            {
                double delta = learningRate*_oGrads[i]*1.0;
                _hoBiases[i] += delta;
                _hoBiases[i] += momentum*_hoPrevBiasesDelta[i];
                _hoPrevBiasesDelta[i] = delta;
            }
        }

        // UpdateWeights

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (_numInput*_numHidden) + (_numHidden*_numOutput) + _numHidden + _numOutput;
            if (weights.Length != numWeights)
                throw new Exception("The weights array length: " + weights.Length +
                                    " does not match the total number of weights and biases: " + numWeights);

            int k = 0; // points into weights param

            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    _ihWeights[i][j] = weights[k++];

            for (int i = 0; i < _numHidden; ++i)
                _ihBiases[i] = weights[k++];

            for (int i = 0; i < _numHidden; ++i)
                for (int j = 0; j < _numOutput; ++j)
                    _hoWeights[i][j] = weights[k++];

            for (int i = 0; i < _numOutput; ++i)
                _hoBiases[i] = weights[k++];
        }

        public double[] GetWeights()
        {
            int numWeights = (_numInput*_numHidden) + (_numHidden*_numOutput) + _numHidden + _numOutput;
            var result = new double[numWeights];
            int k = 0;
            foreach (var t in _ihWeights)
                for (int j = 0; j < _ihWeights[0].Length; ++j)
                    result[k++] = t[j];
            foreach (double t in _ihBiases)
                result[k++] = t;
            foreach (var t in _hoWeights)
                for (int j = 0; j < _hoWeights[0].Length; ++j)
                    result[k++] = t[j];
            foreach (double t in _hoBiases)
                result[k++] = t;
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != _numInput)
                throw new Exception("Inputs array length " + _inputs.Length + " does not match NN numInput value " +
                                    _numInput);

            for (int i = 0; i < _numHidden; ++i)
                _ihSums[i] = 0.0;
            for (int i = 0; i < _numOutput; ++i)
                _hoSums[i] = 0.0;

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                _inputs[i] = xValues[i];

            for (int j = 0; j < _numHidden; ++j) // compute input-to-hidden weighted sums
                for (int i = 0; i < _numInput; ++i)
                    _ihSums[j] += _inputs[i]*_ihWeights[i][j];

            for (int i = 0; i < _numHidden; ++i) // add biases to input-to-hidden sums
                _ihSums[i] += _ihBiases[i];

            for (int i = 0; i < _numHidden; ++i) // determine input-to-hidden output
                _ihOutputs[i] = _activation.Activate(_ihSums[i]);

            for (int j = 0; j < _numOutput; ++j) // compute hidden-to-output weighted sums
                for (int i = 0; i < _numHidden; ++i)
                    _hoSums[j] += _ihOutputs[i]*_hoWeights[i][j];

            for (int i = 0; i < _numOutput; ++i) // add biases to input-to-hidden sums
                _hoSums[i] += _hoBiases[i];

            for (int i = 0; i < _numOutput; ++i) // determine hidden-to-output result
                _outputs[i] = _activation.Activate(_hoSums[i]);

            var result = new double[_numOutput]; // could define a GetOutputs method instead
            _outputs.CopyTo(result, 0);

            return result;
        }

        private void ComputeHiddenGradients()
        {
            for (int i = 0; i < _hGrads.Length; ++i)
            {
                double derivative = (1 - _ihOutputs[i])*_ihOutputs[i];
                // (1 / 1 + exp(-x))'  -- using output value of neuron
                double sum = 0.0;
                for (int j = 0; j < _numOutput; ++j) // each hidden delta is the sum of numOutput terms
                    sum += _oGrads[j]*_hoWeights[i][j]; // each downstream gradient * outgoing weight
                _hGrads[i] = derivative*sum;
            }
        }

        private void ComputeGradientOutput(double[] tValues)
        {
            for (int i = 0; i < _oGrads.Length; ++i)
            {
                double derivative = (1 - _outputs[i])*(1 + _outputs[i]); // derivative of tanh
                _oGrads[i] = derivative*(tValues[i] - _outputs[i]);
            }
        }
    }
}