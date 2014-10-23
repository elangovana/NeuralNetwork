using System.Collections.Generic;
using System.Linq;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public class RankBasedSelector : ISelector
    {
        public IEnumerable<AbstractNetwork> SelectFittestNetworks(AbstractNetwork[] networks, double[] scores, int n)
        {
           return scores.Select((r, i) => new {Network = networks[i], Score = r})
                  .OrderByDescending(x => x.Score)
                  .Take(n)
                  .Select(x => x.Network);
        }
    }
}