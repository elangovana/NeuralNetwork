using System.Runtime.Serialization;

namespace AE.MachineLearning.NeuralNet.Core
{
    [DataContract]
    public class Neuron
    {
        public IActivation Activation { get; set; }

        public Neuron(IActivation activation)
        {
            Activation = activation;
        }

        [DataMember]
        public double[] Weights { get; set; }

        [DataMember]
        public double Bias { get; set; }

        [DataMember]
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

            Output = Activation.CalculateActivate(result);
        }
    }
}