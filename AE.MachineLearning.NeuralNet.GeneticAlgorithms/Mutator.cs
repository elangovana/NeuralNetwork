using System;
using System.Collections.Generic;
using System.Linq;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public class Mutator : IMutator
    {
        public Mutator(INetworkFactory networkFactory, int minNodes, int maxNode, int mutationSize)
        {
            NetworkFactory = networkFactory;
            MinNodes = minNodes;
            MaxNode = maxNode;
            MutationSize = mutationSize;
        }

        public INetworkFactory NetworkFactory { get; set; }
        public int MinNodes { get; set; }
        public int MaxNode { get; set; }
        public int MutationSize { get; set; }

        public AbstractNetwork[] Mutate(List<AbstractNetwork> parentNetworks, double mutationRate)
        {
            var noOfMutants = (int) Math.Floor(parentNetworks.Count*mutationRate);

            var mutatantIndices = new List<int>();
            var rand = new Random();
            var mutantnetworks = new AbstractNetwork[noOfMutants];
            for (int i = 0; i < noOfMutants; i++)
            {
                int indexToMutate;
                do
                {
                    indexToMutate = rand.Next(0, parentNetworks.Count);
                } while (mutatantIndices.Any(x => x == indexToMutate));

                mutatantIndices.Add(indexToMutate);

                var addOrDeleteNode = rand.Next(0, 2);
                var layerToChange = rand.Next(0, parentNetworks[indexToMutate].NetworkLayers.Length - 2);

                NetworkFactory.Activation = parentNetworks[indexToMutate].Activation;
                NetworkFactory.ActivationOutput = parentNetworks[indexToMutate].ActivationOutput;
                NetworkFactory.NumberOfHiddenLayers = parentNetworks[indexToMutate].NumberOfHiddenLayers;
                NetworkFactory.NumberOfInputFeatures = parentNetworks[indexToMutate].NumberOfInputFeatures;
                NetworkFactory.NumberOfOutputs = parentNetworks[indexToMutate].NumberOfOutputs;
                NetworkFactory.NumberOfneuronsForHiddenLayers = new int[parentNetworks[indexToMutate].NumberOfneuronsForHiddenLayers.Length];
                for (int j = 0; j < parentNetworks[indexToMutate].NumberOfneuronsForHiddenLayers.Length; j++)
                {
                    NetworkFactory.NumberOfneuronsForHiddenLayers[j] =
                        parentNetworks[indexToMutate].NumberOfneuronsForHiddenLayers[j];
                }

                var hasChanged = false;
                if (addOrDeleteNode == 0)
                {
                    if (NetworkFactory.NumberOfneuronsForHiddenLayers[layerToChange] + MutationSize > MaxNode)
                        addOrDeleteNode = 1;
                    else
                    {
                        hasChanged = true;
                        NetworkFactory.NumberOfneuronsForHiddenLayers[layerToChange] += MutationSize;
                    }
                }

                if (addOrDeleteNode == 1)
                {
                    if (NetworkFactory.NumberOfneuronsForHiddenLayers[layerToChange] - MutationSize > MinNodes)
                    {
                        hasChanged = true;
                        NetworkFactory.NumberOfneuronsForHiddenLayers[layerToChange] -= MutationSize;
                    }
                }
               
                if (!hasChanged)
                {
                    NetworkFactory.NumberOfneuronsForHiddenLayers[layerToChange] = rand.Next(MinNodes, MaxNode + 1);
                }

                mutantnetworks[i] = NetworkFactory.CreateNetwork();
            }

            return mutantnetworks;
        }
    }
}