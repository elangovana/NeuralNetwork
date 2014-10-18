using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.HandWrittenDigits.App
{
    public class HandwrittenDigitRecogniser
    {
       public static void Calculate(string trainFile, string testFile)
        {
            var data = new HandandWrittenDataLoader();
            data.LoadData(trainFile, testFile);

            var netWork = new NeuralNetwork(data.Inputs[0].Length, data.Outputs[0].Length, 1, new int[] {100},
                                            new HyperTanActivation());

            netWork.InitNetworkWithRandomWeights();

            new BackPropagationTraining(netWork, new SquaredCostFunction()).Train(data.Inputs, data.Outputs, .9, 0.0);
        }
    }
}