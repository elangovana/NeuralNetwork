using System;
using System.Collections.Generic;
using System.Linq;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public class Mutator : IMutator
    {
        public Mutator(INetworkFactory networkFactory)
        {
            NetworkFactory = networkFactory;
        }

        public INetworkFactory NetworkFactory { get; set; }

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
                NetworkFactory.NumberOfneuronsForHiddenLayers = parentNetworks[indexToMutate].NumberOfneuronsForHiddenLayers;
                NetworkFactory.NumberOfneuronsForHiddenLayers[layerToChange] = addOrDeleteNode == 0 || NetworkFactory
                                                                                         .NumberOfneuronsForHiddenLayers
                                                                                         [layerToChange] ==1
                                                                                   ? NetworkFactory
                                                                                         .NumberOfneuronsForHiddenLayers
                                                                                         [layerToChange] + 1
                                                                                   : NetworkFactory
                                                                                         .NumberOfneuronsForHiddenLayers
                                                                                         [layerToChange] - 1;
                mutantnetworks[i] = NetworkFactory.CreateNetwork();
            }

            return mutantnetworks;
        }
    }
}