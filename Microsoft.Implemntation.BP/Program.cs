using System;

namespace BackPropagation
{
    class BackPropagationProgram
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("\nBegin Neural Network Back-Propagation demo\n");

                Console.WriteLine("Creating a 3-input, 4-hidden, 2-output neural network");
                Console.WriteLine("Using sigmoid function for input-to-hidden activation");
                Console.WriteLine("Using tanh function for hidden-to-output activation");
                NeuralNetwork nn = new NeuralNetwork(3, 4, 2);

                // arbitrary weights and biases
                double[] weights = new double[] {
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
          -2.0, -6.0, -1.0, -7.0,
          1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
          -2.5, -5.0 };

                Console.WriteLine("\nInitial 26 random weights and biases are:");
                Helpers.ShowVector(weights, 2, true);

                Console.WriteLine("Loading neural network weights and biases");
                nn.SetWeights(weights);

                Console.WriteLine("\nSetting inputs:");
                double[] xValues = new double[] { 1.0, 2.0, 3.0 };
                Helpers.ShowVector(xValues, 2, true);

                double[] initialOutputs = nn.ComputeOutputs(xValues);
                Console.WriteLine("Initial outputs:");
                Helpers.ShowVector(initialOutputs, 4, true);

                double[] tValues = new double[] { -0.8500, 0.7500 }; // target (desired) values. note these only make sense for tanh output activation
                Console.WriteLine("Target outputs to learn are:");
                Helpers.ShowVector(tValues, 4, true);

                double eta = 0.90;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
                double alpha = 0.00; // momentum - to discourage oscillation. found by trial and error.
                Console.WriteLine("Setting learning rate (eta) = " + eta.ToString("F2") + " and momentum (alpha) = " + alpha.ToString("F2"));

                Console.WriteLine("\nEntering main back-propagation compute-update cycle");
                Console.WriteLine("Stopping when sum absolute error <= 0.01 or 1,000 iterations\n");
                int ctr = 0;
                double[] yValues = nn.ComputeOutputs(xValues); // prime the back-propagation loop
                double error = Error(tValues, yValues);
                while (ctr < 1000 && error > 0.01)
                {
                    Console.WriteLine("===================================================");
                    Console.WriteLine("iteration = " + ctr);
                    Console.WriteLine("Updating weights and biases using back-propagation");
                    nn.UpdateWeights(tValues, eta, alpha);
                    Console.WriteLine("Computing new outputs:");
                    yValues = nn.ComputeOutputs(xValues);
                    Helpers.ShowVector(yValues, 4, false);
                    Console.WriteLine("\nComputing new error");
                    error = Error(tValues, yValues);
                    Console.WriteLine("Error = " + error.ToString("F4"));
                    ++ctr;
                }
                Console.WriteLine("===================================================");
                Console.WriteLine("\nBest weights and biases found:");
                double[] bestWeights = nn.GetWeights();
                Helpers.ShowVector(bestWeights, 2, true);

                Console.WriteLine("End Neural Network Back-Propagation demo\n");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Fatal: " + ex.Message);
                Console.ReadLine();
            }
        } // Main

        static double Error(double[] target, double[] output) // sum absolute error. could put into NeuralNetwork class.
        {
            double sum = 0.0;
            for (int i = 0; i < target.Length; ++i)
                sum += Math.Abs(target[i] - output[i]);
            return sum;
        }

    } // class BackPropagation

    class NeuralNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights; // input-to-hidden
        private double[] ihSums;
        private double[] ihBiases;
        private double[] ihOutputs;

        private double[][] hoWeights;  // hidden-to-output
        private double[] hoSums;
        private double[] hoBiases;
        private double[] outputs;

        private double[] oGrads; // output gradients for back-propagation
        private double[] hGrads; // hidden gradients for back-propagation

        private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
        private double[] ihPrevBiasesDelta;

        private double[][] hoPrevWeightsDelta;
        private double[] hoPrevBiasesDelta;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];
            ihWeights = Helpers.MakeMatrix(numInput, numHidden);
            ihSums = new double[numHidden];
            ihBiases = new double[numHidden];
            ihOutputs = new double[numHidden];
            hoWeights = Helpers.MakeMatrix(numHidden, numOutput);
            hoSums = new double[numOutput];
            hoBiases = new double[numOutput];
            outputs = new double[numOutput];

            oGrads = new double[numOutput];
            hGrads = new double[numHidden];

            ihPrevWeightsDelta = Helpers.MakeMatrix(numInput, numHidden);
            ihPrevBiasesDelta = new double[numHidden];
            hoPrevWeightsDelta = Helpers.MakeMatrix(numHidden, numOutput);
            hoPrevBiasesDelta = new double[numOutput];
        }

        public void UpdateWeights(double[] tValues, double eta, double alpha) // update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
        {
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and matrices have values (other than 0.0)
            if (tValues.Length != numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. compute output gradients
            for (int i = 0; i < oGrads.Length; ++i)
            {
                double derivative = (1 - outputs[i]) * (1 + outputs[i]); // derivative of tanh
                oGrads[i] = derivative * (tValues[i] - outputs[i]);
            }

            // 2. compute hidden gradients
            for (int i = 0; i < hGrads.Length; ++i)
            {
                double derivative = (1 - ihOutputs[i]) * ihOutputs[i]; // (1 / 1 + exp(-x))'  -- using output value of neuron
                double sum = 0.0;
                for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
                    sum += oGrads[j] * hoWeights[i][j]; // each downstream gradient * outgoing weight
                hGrads[i] = derivative * sum;
            }

            // 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order
            for (int i = 0; i < ihWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < ihWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = eta * hGrads[j] * inputs[i]; // compute the new delta
                    ihWeights[i][j] += delta; // update
                    ihWeights[i][j] += alpha * ihPrevWeightsDelta[i][j]; // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                }
            }

            // 3b. update input to hidden biases
            for (int i = 0; i < ihBiases.Length; ++i)
            {
                double delta = eta * hGrads[i] * 1.0; // the 1.0 is the constant input for any bias; could leave out
                ihBiases[i] += delta;
                ihBiases[i] += alpha * ihPrevBiasesDelta[i];
            }

            // 4. update hidden to output weights
            for (int i = 0; i < hoWeights.Length; ++i)  // 0..3 (4)
            {
                for (int j = 0; j < hoWeights[0].Length; ++j) // 0..1 (2)
                {
                    double delta = eta * oGrads[j] * ihOutputs[i];  // see above: ihOutputs are inputs to next layer
                    hoWeights[i][j] += delta;
                    hoWeights[i][j] += alpha * hoPrevWeightsDelta[i][j];
                    hoPrevWeightsDelta[i][j] = delta;
                }
            }

            // 4b. update hidden to output biases
            for (int i = 0; i < hoBiases.Length; ++i)
            {
                double delta = eta * oGrads[i] * 1.0;
                hoBiases[i] += delta;
                hoBiases[i] += alpha * hoPrevBiasesDelta[i];
                hoPrevBiasesDelta[i] = delta;
            }
        } // UpdateWeights

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("The weights array length: " + weights.Length + " does not match the total number of weights and biases: " + numWeights);

            int k = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                ihBiases[i] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];

            for (int i = 0; i < numOutput; ++i)
                hoBiases[i] = weights[k++];
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < ihBiases.Length; ++i)
                result[k++] = ihBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < hoBiases.Length; ++i)
                result[k++] = hoBiases[i];
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Inputs array length " + inputs.Length + " does not match NN numInput value " + numInput);

            for (int i = 0; i < numHidden; ++i)
                ihSums[i] = 0.0;
            for (int i = 0; i < numOutput; ++i)
                hoSums[i] = 0.0;

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)  // compute input-to-hidden weighted sums
                for (int i = 0; i < numInput; ++i)
                    ihSums[j] += this.inputs[i] * ihWeights[i][j];

            for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
                ihSums[i] += ihBiases[i];

            for (int i = 0; i < numHidden; ++i)   // determine input-to-hidden output
                ihOutputs[i] = SigmoidFunction(ihSums[i]);

            for (int j = 0; j < numOutput; ++j)   // compute hidden-to-output weighted sums
                for (int i = 0; i < numHidden; ++i)
                    hoSums[j] += ihOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                hoSums[i] += hoBiases[i];

            for (int i = 0; i < numOutput; ++i)   // determine hidden-to-output result
                this.outputs[i] = HyperTanFunction(hoSums[i]);

            double[] result = new double[numOutput]; // could define a GetOutputs method instead
            this.outputs.CopyTo(result, 0);

            return result;
        } // ComputeOutputs

        private static double StepFunction(double x) // an activation function that isn't compatible with back-propagation bcause it isn't differentiable
        {
            if (x > 0.0) return 1.0;
            else return 0.0;
        }

        private static double SigmoidFunction(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            else return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double HyperTanFunction(double x)
        {
            if (x < -10.0) return -1.0;
            else if (x > 10.0) return 1.0;
            else return Math.Tanh(x);
        }
    } // class NeuralNetwork

    // ===========================================================================

    public class Helpers
    {
        public static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        public static void ShowVector(double[] vector, int decimals, bool blankLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % 12 == 0) // max of 12 values per row 
                    Console.WriteLine("");
                if (vector[i] >= 0.0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " "); // n decimals
            }
            if (blankLine) Console.WriteLine("\n");
        }

        public static void ShowMatrix(double[][] matrix, int numRows, int decimals)
        {
            int ct = 0;
            if (numRows == -1) numRows = int.MaxValue; // if numRows == -1, show all rows
            for (int i = 0; i < matrix.Length && ct < numRows; ++i)
            {
                for (int j = 0; j < matrix[0].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" "); // blank space instead of '+' sign
                    Console.Write(matrix[i][j].ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
                ++ct;
            }
            Console.WriteLine("");
        }

    } // class Helpers

} // ns
