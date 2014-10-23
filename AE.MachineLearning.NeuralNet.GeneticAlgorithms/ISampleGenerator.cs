using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    internal interface ISampleGenerator
    {
        AbstractNetwork[] SampleNetworkPopulation(int minLayer, int maxLayer, int minNodes, int maxNodes,
                                                  int sampleSize, int? seed = null);
    }
}