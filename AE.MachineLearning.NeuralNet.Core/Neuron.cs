namespace AE.MachineLearning.NeuralNet.Core
{
    public class Neuron
    {
        private readonly IActivation _activation;

        public Neuron(IActivation activation)
        {
            _activation = activation;
        }

        public double[] Weights { get; set; }

        public double Bias { get; set; }

        public double Output { get; private set; }

        public void CalculateOutput(double[] input)
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

            Output = _activation.CalculateActivate(result);
        }
    }
}