namespace AE.MachineLearning.NeuralNet.Core
{
    public class Neuron
    {
        public double[] Weights { get; set; }

        public double Bias { get; set; }

        public double CalculateOutput(double[] input, IActivation activation)
        {
            double result = 0;

            //Validate
            if (input.Length != Weights.Length)
                throw new NeuralNetException(
                    string.Format("The numbers weights {0} of do not match the number of inputs{1}", Weights.Length,
                                  input.Length));

            for (int i = 0; i < input.Length; i++)
            {
                result = result + input[i]*Weights[i];
            }

            result = result + Bias;

            return activation.CalculateActivate(result);
        }
    }
}