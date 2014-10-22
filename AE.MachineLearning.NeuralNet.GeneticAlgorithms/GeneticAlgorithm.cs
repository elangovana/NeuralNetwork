using System;
using System.IO;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public class GeneticAlgorithm : IGeneticAlgorithm
    {
        private readonly IFitnessCalculator _fitnessCalculator;

        private readonly int _maxlayers;
        private readonly int _minLayer;
        private readonly INetworkFactory _networkFactory;
        private readonly ITrainingAlgoritihm _trainingAlgoritihm;
        private int _flushCounter;
        private int _numOfInputs;

        private int _numOfOutputs;
        private int _sampleSize = 10;
        private readonly Sampler _sampler;

        public GeneticAlgorithm(int numOfInputs, int numOfOutputs, int minLayer, int maxlayers,
                                IFitnessCalculator fitnessCalculator, ITrainingAlgoritihm trainingAlgoritihm,
                                INetworkFactory networkFactory)
        {
            _numOfInputs = numOfInputs;
            _numOfOutputs = numOfOutputs;
            _minLayer = minLayer;
            _maxlayers = maxlayers;
            _fitnessCalculator = fitnessCalculator;
            _trainingAlgoritihm = trainingAlgoritihm;
            _networkFactory = networkFactory;
            _networkFactory.NumberOfInputFeatures = numOfInputs;
            _networkFactory.NumberOfOutputs = numOfOutputs;
            _sampler = new Sampler {NetworkFactory = networkFactory};
        }

        public StreamWriter LogWriter { get; set; }

        private void WriteLog(string message)
        {
            if (LogWriter == null) return;

            LogWriter.WriteLine("{0} - {1}", DateTime.Now, message);
            if (_flushCounter%1 == 0)
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
            AbstractNetwork[] samples = _sampler.SampleNetworkPopulation(_minLayer, _maxlayers, 1, 100, _sampleSize);

            double optimumScore = 0.0;
            AbstractNetwork optimumNetwork = null;
            foreach (AbstractNetwork network in samples)
            {
                LogNetworkDetails(network);
                network.InitNetworkWithRandomWeights();
                _trainingAlgoritihm.Network = network;

                _trainingAlgoritihm.Train(trainInputs, trainOutputs, .2, .7,.8,100);

                double[][] actualOutput = _trainingAlgoritihm.Predict(testInputs);

                double score = _fitnessCalculator.Calculator(testOutputs, actualOutput);
               
                

                if (score > optimumScore)
                {
                    optimumScore = score;
                    optimumNetwork = network;
                }

               
                WriteLog(string.Format("The score for network is {0}", score.ToString("F4")));
            }

            return optimumNetwork;
        }

       public void LogNetworkDetails(AbstractNetwork network)
       {
           WriteLog(string.Format("Network with {0} hidden layers", network.NumberOfHiddenLayers));

           for (int index = 1; index < network.NetworkLayers.Length -1; index++)
           {
               var layer = network.NetworkLayers[index];
               WriteLog(string.Format("Hidden Layer {0}, neurons {1}", index, layer.Neurons.Length));
           }
       }

        private void InitParams(double[][] inputs, double[][] outputs)
        {
            _numOfInputs = inputs[0].Length;
            _numOfOutputs = outputs[0].Length;
        }
    }
}