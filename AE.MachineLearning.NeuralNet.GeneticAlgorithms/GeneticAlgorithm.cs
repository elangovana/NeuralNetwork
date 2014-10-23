using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public class GeneticAlgorithm : IGeneticAlgorithm
    {
        private readonly IFitnessCalculator _fitnessCalculator;

        private int _flushCounter;
        private int _maxNodes = 120;
        private int _minNodes = 1;
        private int _numOfInputs;

        private int _numOfOutputs;
        private int _numberOfGenerations = 10;
        private int _sampleSize = 10000;
        private Sampler _sampler;

        public GeneticAlgorithm(int numOfInputs, int numOfOutputs, int minLayer, int maxlayers,
                                IFitnessCalculator fitnessCalculator, ITrainingAlgoritihm trainingAlgoritihm,
                                INetworkFactory networkFactory, ISelector selector)
        {
            _numOfInputs = numOfInputs;
            _numOfOutputs = numOfOutputs;
            MinLayer = minLayer;
            Maxlayers = maxlayers;
            _fitnessCalculator = fitnessCalculator;
            TrainingAlgoritihm = trainingAlgoritihm;
            NetworkFactory = networkFactory;
            Selector = selector;
            NetworkFactory.NumberOfInputFeatures = numOfInputs;
            NetworkFactory.NumberOfOutputs = numOfOutputs;
            _sampler = new Sampler {NetworkFactory = networkFactory};
        }

        public int MaxNodes
        {
            get { return _maxNodes; }
            set { _maxNodes = value; }
        }

        public int MinNodes
        {
            get { return _minNodes; }
            set { _minNodes = value; }
        }

        public int SampleSize
        {
            get { return _sampleSize; }
            set { _sampleSize = value; }
        }

        public int Maxlayers { get; set; }

        public int MinLayer { get; set; }

        public INetworkFactory NetworkFactory { get; set; }

      

        public ITrainingAlgoritihm TrainingAlgoritihm { get; set; }

        public ISelector Selector { get; set; }
        public StreamWriter LogWriter { get; set; }

        public int NumberOfGenerations
        {
            get { return _numberOfGenerations; }
            set { _numberOfGenerations = value; }
        }

        public AbstractNetwork Optimise(double[][] trainInputs, double[][] trainOutputs, double[][] testInputs,
                                        double[][] testOutputs)
        {
            InitParams(trainInputs, trainOutputs);

            AbstractNetwork[] samples = _sampler.SampleNetworkPopulation(MinLayer, Maxlayers, _minNodes, _maxNodes,
                                                                        SampleSize);


            int igen = 0;
            LogSettings();
            var scores = new double[samples.Length];
            do
            {
                WriteLog(string.Format("--------------------------- gen{0}", igen));

                CalculateFitness(trainInputs, trainOutputs, testInputs, testOutputs, samples, scores);
                int n = samples.Length/2;
                IEnumerable<AbstractNetwork> fittestSamples = Selector.SelectFittestNetworks(samples, scores, n);
                AbstractNetwork[] newSamples = _sampler.SampleNetworkPopulation(MinLayer, Maxlayers, _minNodes,
                                                                               _maxNodes, samples.Length - n);
                samples = MergeSamples(fittestSamples, newSamples);
                igen++;
            } while (igen <= NumberOfGenerations);


            return Selector.SelectFittestNetworks(samples, scores, 1).First();
        }

        private void LogSettings()
        {
            WriteLog(string.Format("Population Size {0}, Max Generations {1}, Min Layer {2}, Max Layer {3}, Min Nodes {4}, Max Nodes {5}", SampleSize, NumberOfGenerations, MinLayer, Maxlayers, MinNodes, MaxNodes));
        }

        private AbstractNetwork[] MergeSamples(IEnumerable<AbstractNetwork> fittestSamples, AbstractNetwork[] newSamples)
        {
            IEnumerable<AbstractNetwork> abstractNetworks = fittestSamples as AbstractNetwork[] ??
                                                            fittestSamples.ToArray();
            var networks = new AbstractNetwork[abstractNetworks.Count() + newSamples.Length];
            int i = 0;
            foreach (AbstractNetwork abstractNetwork in abstractNetworks)
            {
                networks[i] = abstractNetwork;
                i++;
            }

            foreach (AbstractNetwork abstractNetwork in newSamples)
            {
                networks[i] = abstractNetwork;
                i++;
            }

            return networks;
        }

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

        private void CalculateFitness(double[][] trainInputs, double[][] trainOutputs, double[][] testInputs,
                                      double[][] testOutputs, AbstractNetwork[] samples, double[] scores)
        {
            for (int s = 0; s < samples.Length; s++)
            {
                AbstractNetwork network = samples[s];
                LogNetworkDetails(network);
                network.InitNetworkWithRandomWeights();
                TrainingAlgoritihm.Network = network;

                TrainingAlgoritihm.Train(trainInputs, trainOutputs);

                double[][] actualOutput = TrainingAlgoritihm.Predict(testInputs);

                double score = _fitnessCalculator.Calculator(testOutputs, actualOutput);

                scores[s] = score;


                WriteLog(
                    string.Format(
                        "The score for network with {0} hidden layers, with avg no nuerons {1} is {2}",
                        network.NumberOfHiddenLayers, GetAverageNeuronsPerLayer(network).ToString("F4"),
                        score.ToString("F4")));
            }
        }

        private void LogNetworkDetails(AbstractNetwork network)
        {
            WriteLog(string.Format("Network with {0} hidden layers", network.NumberOfHiddenLayers));

            for (int index = 1; index < network.NetworkLayers.Length - 1; index++)
            {
                NetworkLayer layer = network.NetworkLayers[index];
                WriteLog(string.Format("Hidden Layer {0}, neurons {1}", index, layer.Neurons.Length));
            }
        }

        private double GetAverageNeuronsPerLayer(AbstractNetwork network)
        {
            int total = 0;
            for (int i = 1; i < network.NetworkLayers.Length - 1; i++)
            {
                total += network.NetworkLayers[i].NumOfNeurons;
            }

            return ((double) total/network.NumberOfHiddenLayers);
        }


        private void InitParams(double[][] inputs, double[][] outputs)
        {
            _numOfInputs = inputs[0].Length;
            _numOfOutputs = outputs[0].Length;
        }
    }
}