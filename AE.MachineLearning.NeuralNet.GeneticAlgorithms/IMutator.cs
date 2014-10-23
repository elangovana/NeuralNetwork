using System;
using System.Collections.Generic;
using System.Linq;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public interface IMutator
    {
        List<AbstractNetwork> Mutate(List<AbstractNetwork> networks, double mutationRate);
    }

    public class Mutator : IMutator
    {
        public List<AbstractNetwork> Mutate(List<AbstractNetwork> networks, double mutationRate)
        {
            var noOfMutants = (int) Math.Ceiling(networks.Count*mutationRate);

            var mutatantIndices = new List<int>();
            var rand = new Random();
            for (int i = 0; i < noOfMutants; i++)
            {
                int indexToMutate;
                do
                {
                    indexToMutate = rand.Next(0, networks.Count - 1);
                } while (mutatantIndices.Any(x => x == indexToMutate));

                mutatantIndices.Add(indexToMutate);

                var addOrDeleteNode = rand.Next(0, 1);
                var layerToChange = rand.Next(1, networks[indexToMutate].NetworkLayers.Length - 2);

               
            }

            throw new NotImplementedException();
        }
    }
}