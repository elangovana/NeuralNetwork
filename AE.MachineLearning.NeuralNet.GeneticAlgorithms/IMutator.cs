using System.Collections.Generic;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public interface IMutator
    {
        AbstractNetwork[] Mutate(List<AbstractNetwork> parentNetworks, double mutationRate);
    }
}