using System;
using System.Collections.Generic;
using System.IO;

namespace AE.MachineLearning.NeuralNet.Core
{
    internal class GeneticAlgorithm
    {
        private readonly IFitnessCalculator _fitnessCalculator;

        private readonly int _maxlayers;
        private readonly int _minLayer;
        private readonly ITrainingAlgoritihm _trainingAlgoritihm;
        private readonly INetworkFactory _networkFactory;
        private int _flushCounter;
        private int _numOfInputs;

        private int _numOfOutputs;
        private int _sampleSize = 10;

        public GeneticAlgorithm(int numOfInputs, int numOfOutputs, int minLayer, int maxlayers,
                                IFitnessCalculator fitnessCalculator, ITrainingAlgoritihm trainingAlgoritihm, INetworkFactory networkFactory)
        {
            _numOfInputs = numOfInputs;
            _numOfOutputs = numOfOutputs;
            _minLayer = minLayer;
            _maxlayers = maxlayers;
            _fitnessCalculator = fitnessCalculator;
            _trainingAlgoritihm = trainingAlgoritihm;
            _networkFactory = networkFactory;
        }

        public StreamWriter LogWriter { get; set; }

        private void WriteLog(string message)
        {
            if (LogWriter == null) return;

            LogWriter.WriteLine("{0} - {1}", DateTime.Now, message);
            if (_flushCounter%10 == 0)
            {
                LogWriter.Flush();
                _flushCounter = 0;
            }
            _flushCounter++;
        }

        public AbstractNetwork Optimise(double[][] trainInputs, double[][] trainOutputs, double[][] testInputs,
                                      double[][] testOutputs)
        {
            InitParams(trainInputs, trainOutputs);
            AbstractNetwork[] samples = SampleNetworkPopulation(_sampleSize);

            double optimumScore = 0.0;
            AbstractNetwork optimumNetwork = null;
            for (int i = 0; i < samples.Length; i++)
            {
                _trainingAlgoritihm.Train(trainInputs, trainOutputs, .01, .7);

                double[][] actualOutput = _trainingAlgoritihm.Predict(testInputs);

                double score = _fitnessCalculator.Calculator(testInputs, testOutputs);

                if (score > optimumScore)
                {
                    optimumScore = score;
                    optimumNetwork = samples[i];
                }
            }

            return optimumNetwork;
        }

        private AbstractNetwork[] SampleNetworkPopulation(int sampleSize, int? seed = null)
        {
            var sampledNetworks = new List<AbstractNetwork>();
            Random random = seed == null ? new Random() : new Random(seed.Value);
            for (int i = 0; i < sampleSize; i++)
            {
                int numOfHiddenLayers = random.Next(_minLayer, _maxlayers);

                var numOfNodesPerHiddenLayer = new int[numOfHiddenLayers];

               
                for (int j = 0; j < numOfHiddenLayers; j++)
                {
                    numOfNodesPerHiddenLayer[j] = random.Next(1, 100);
                }

                _networkFactory.NumberOfHiddenLayers = numOfHiddenLayers;
                _networkFactory.NumberOfneuronsForHiddenLayers = numOfNodesPerHiddenLayer;
                sampledNetworks.Add(_networkFactory.CreateNetwork());
            }

            return sampledNetworks.ToArray();
        }

        private void InitParams(double[][] inputs, double[][] outputs)
        {
            _numOfInputs = inputs[0].Length;
            _numOfOutputs = outputs[0].Length;
        }
    }
}