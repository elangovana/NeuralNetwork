using System;

namespace AE.MachineLearning.NeuralNet.Core
{
    internal class GeneticAlgorithm
    {
        private readonly int _maxlayers;
        private readonly int _minLayer;
        private int _numOfInputs;

        private int _numOfOutputs;

        public GeneticAlgorithm(int minLayer, int maxlayers)
        {
            _minLayer = minLayer;
            _maxlayers = maxlayers;
        }


        public NetworkLayer Train(double[][] inputs, double[][] outputs)
        {
            InitParams(inputs, outputs);


             SampleNetworkPopulation();
              
            throw  new NotImplementedException();
        }

        private void SampleNetworkPopulation()
        {
          
        }

        private void InitParams(double[][] inputs, double[][] outputs)
        {
            _numOfInputs = inputs[0].Length;
            _numOfOutputs = outputs[0].Length;
        }
    }
}