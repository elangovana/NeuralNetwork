using System;
using System.Collections.Generic;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    internal class Sampler
    {
        public INetworkFactory NetworkFactory { get; set; }

        public AbstractNetwork[] SampleNetworkPopulation(int minLayer, int maxLayer, int minNodes, int maxNodes,
                                                         int sampleSize, int? seed = null)
        {
            var sampledNetworks = new List<AbstractNetwork>();
            Random random = seed == null ? new Random() : new Random(seed.Value);
            for (int i = 0; i < sampleSize; i++)
            {
                int numOfHiddenLayers = random.Next(minLayer, maxLayer + 1);

                var numOfNodesPerHiddenLayer = new int[numOfHiddenLayers];


                for (int j = 0; j < numOfHiddenLayers; j++)
                {
                    numOfNodesPerHiddenLayer[j] = random.Next(minNodes, maxNodes + 1);
                }


                AbstractNetwork network = CreateNetwork(numOfHiddenLayers, numOfNodesPerHiddenLayer);
                sampledNetworks.Add(network);
            }

            return sampledNetworks.ToArray();
        }

        private AbstractNetwork CreateNetwork(int numOfHiddenLayers, int[] numOfNodesPerHiddenLayer)
        {
            NetworkFactory.NumberOfHiddenLayers = numOfHiddenLayers;
            NetworkFactory.NumberOfneuronsForHiddenLayers = numOfNodesPerHiddenLayer;
          
            AbstractNetwork network = NetworkFactory.CreateNetwork();
            return network;
        }
    }
}