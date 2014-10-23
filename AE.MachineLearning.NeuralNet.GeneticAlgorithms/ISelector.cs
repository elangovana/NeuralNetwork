using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public interface ISelector
    {
        IEnumerable<AbstractNetwork> SelectFittestNetworks(AbstractNetwork[] networks, double[] scores, int n);
    }
}
