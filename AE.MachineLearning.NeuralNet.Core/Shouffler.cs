using System;
using System.Linq;

namespace AE.MachineLearning.NeuralNet.Core
{
    public class Shouffler : IShouffler
    {
        public void Shouffle(double[][] inputs, double[][] outputs, out double[][] randomisedInputs,
                             out double[][] randomisedOutputs)
        {
            var random = new Random();
            var randomOrder = new double[inputs.Length];
            int randomLimit = inputs.Length*2;
            for (int i = 0; i < randomOrder.Length; i++)
            {
                randomOrder[i] = random.Next(0, randomLimit);
            }

            var randomised = inputs.Select((r, i) => new {input = r, output = outputs[i], order = randomOrder[i]})
                                   .OrderBy(x => x.order).ToArray();

            randomisedInputs = new double[inputs.Length][];
            randomisedOutputs = new double[outputs.Length][];
            for (int i = 0; i < randomised.Count(); i++)
            {
                randomisedInputs[i] = randomised[i].input;
                randomisedOutputs[i] = randomised[i].output;
            }
        }
    }
}