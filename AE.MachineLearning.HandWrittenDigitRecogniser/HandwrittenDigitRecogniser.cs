using System;
using System.IO;
using AE.MachineLearning.NeuralNet.Core;
using AE.MachineLearning.NeuralNet.GeneticAlgorithms;

namespace AE.MachineLearning.HandWrittenDigitRecogniser
{
    public class HandwrittenDigitRecogniser : IDisposable
    {
        private readonly double _learningRate;
        private readonly double _momentum;
        private readonly string _networkFile;
        private readonly string _outDir;
        private readonly string _testFile;
        private readonly string _trainFile;
        private bool _isDisposed;
        private StreamWriter _writer;

        public HandwrittenDigitRecogniser(string trainFile, string testFile, string outDir, double learningRate,
                                          double momentum, string networkFile = null)
        {
            _trainFile = trainFile;
            _testFile = testFile;
            _outDir = Path.Combine(outDir, string.Format("Run{0}", DateTime.Now.ToString("yyyyMMddhhmmss")));
            Helper.SetUpDir(_outDir);
            _networkFile = networkFile;
            _learningRate = learningRate;
            _momentum = momentum;
        }

        private StreamWriter RunLogWriter
        {
            get { return _writer ?? (_writer = new StreamWriter(Path.Combine(_outDir, "Runlog.txt"))); }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Run(int maxIteration = 10000, double maxError = .05)
        {
            Writelog("Starting Run");

            var data = new HandandWrittenDataLoader();
            data.LoadData(_trainFile, _testFile);

            AbstractNetwork netWork = CreateNetwork(_networkFile, data);


            //Write Network init
            int i = 1;
            double correctRate = 0.0;
            double previousCorrectRate;
            do
            {
                previousCorrectRate = correctRate;
                string fileName = string.Format("Networkfinal{0}.xml", i);
                BackPropagationTraining trainingAlgoritihmAlgorithm = Train(data, netWork, maxIteration, maxError,
                                                                            fileName);

                i++;
                correctRate = Predict(data, trainingAlgoritihmAlgorithm);
            } while (previousCorrectRate < correctRate);


            Writelog("Procesing complete");
        }

        public void RunGeneticAlgorithm(int minLayers, int maxLayers, int minNoOfNodes, int maxNoOfNodes,
                                        int numberOfGenerations, int populationSize, int iterationPerTraning,
                                        int mutationSize, int maxIteration = 10000, double maxError = .05)
        {
            var data = new HandandWrittenDataLoader();
            data.LoadData(_trainFile, _testFile);

            var feedForwardLayerNeuralNetworkFactory = new FeedForwardLayerNeuralNetworkFactory();
            feedForwardLayerNeuralNetworkFactory.Activation = new HyperTanActivation();
            var selector = new RankBasedSelector();
            var ga = new GeneticAlgorithm(data.Inputs[0].Length, data.Outputs[0].Length, minLayers, maxLayers,
                                          new ClassficationFitnessCalculator(),
                                          new BackPropagationTraining(
                                              new EntropyLossGradientCalc(new HyperTanActivation()))
                                              {
                                                  MaxError = maxError,
                                                  LearningRate = _learningRate,
                                                  Momentum = _momentum,
                                                  MaxIteration = maxIteration,
                                                  LogWriter = RunLogWriter,
                                                  LogLevel = 3
                                              },
                                          feedForwardLayerNeuralNetworkFactory, selector,
                                          new Mutator(feedForwardLayerNeuralNetworkFactory, minNoOfNodes, maxNoOfNodes,
                                                      mutationSize))
                {
                    MinNodes = minNoOfNodes,
                    MaxNodes = maxNoOfNodes,
                    NumberOfGenerations = numberOfGenerations,
                    SampleSize = populationSize,
                    IterationsPerSetting = iterationPerTraning
                };

            RunGa(ga, data.Inputs, data.Outputs);
        }

        public void RunGa(IGeneticAlgorithm ga, double[][] inputs, double[][] outputs)
        {
            //Split data into test and train 70% /30% rule

            var noTests = (int) Math.Floor((inputs.Length*.7));
            int notrain = inputs.Length - noTests;

            var trainInputs = new double[noTests][];
            var trainOutputs = new double[noTests][];
            for (int i = 0; i < noTests; i++)
            {
                trainInputs[i] = inputs[i];
                trainOutputs[i] = outputs[i];
            }

            var testInputs = new double[notrain][];
            var testOutputs = new double[notrain][];
            for (int i = noTests, j = 0; i < inputs.Length; i++ ,j++)
            {
                testInputs[j] = inputs[i];
                testOutputs[j] = outputs[i];
            }
            ga.LogWriter = RunLogWriter;
            AbstractNetwork network = ga.Optimise(trainInputs, trainOutputs, testInputs, testOutputs);
            network.PersistNetwork(Path.Combine(_outDir, "GeFinalNetwork.xml"));
        }


        private double Predict(HandandWrittenDataLoader data, BackPropagationTraining trainingAlgoritihmAlgorithm)
        {
            Writelog(string.Format("Running prediction with test records rows {0} columns {1}", data.TestInputs.Length,
                                   data.TestInputs[0].Length));
            double[][] prediction = trainingAlgoritihmAlgorithm.Predict(data.TestInputs);

            data.WriteData(_testFile, prediction, Path.Combine(_outDir, "predictions.csv"));

            double percentageCorrect = 0.0;
            if (data.GetCorrectTestRate(prediction, out percentageCorrect))
            {
                Writelog(string.Format("Percentage correct prediction {0}", percentageCorrect.ToString("F4")));
            }

            return percentageCorrect;
        }

        private BackPropagationTraining Train(HandandWrittenDataLoader data, AbstractNetwork netWork, int maxIteration,
                                              double maxError, string fileNameToPersistNetwork)
        {
            Writelog(string.Format("Train file Records rows {0} columns {1}", data.Inputs.Length, data.Inputs[0].Length));
            Writelog(
                string.Format(
                    "Begining training using learning rate {0}, momentum {1}, maxIteration {2}, maxError {3}",
                    _learningRate, _momentum, maxIteration, maxError));

            var trainingAlgorithm = new BackPropagationTraining(netWork,
                                                                new EntropyLossGradientCalc(new HyperTanActivation()))
                {
                    LogWriter = RunLogWriter,
                    LearningRate = _learningRate,
                    Momentum = _momentum,
                    MaxError = maxError,
                    MaxIteration = maxIteration,
                    LogLevel = 0
                };

            double[][] randomisedInputs = null;
            double[][] randomisedOutputs = null;
            data.Randomise(data.Inputs, data.Outputs, out randomisedInputs, out randomisedOutputs);
            trainingAlgorithm.Train(randomisedInputs, randomisedOutputs);

            netWork.PersistNetwork(Path.Combine(_outDir, fileNameToPersistNetwork));
            return trainingAlgorithm;
        }


        public void Writelog(string message)
        {
            RunLogWriter.WriteLine("{0} - {1}", DateTime.Now, message);
        }

        private AbstractNetwork CreateNetwork(string networkFile, HandandWrittenDataLoader data)
        {
            AbstractNetwork netWork;
            if (networkFile == null)
            {
                netWork = new NeuralNetwork(data.Inputs[0].Length, data.Outputs[0].Length, 1, new[] {100},
                                            new HyperTanActivation());
                netWork.InitNetworkWithRandomWeights();
            }

            else
            {
                netWork = new NeuralNetwork().LoadNetwork(networkFile, new HyperTanActivation());
            }

            netWork.PersistNetwork(Path.Combine(_outDir, "NetworkInit.xml"));
            return netWork;
        }


        private void Dispose(bool isDisposing)
        {
            if (_isDisposed) return;

            if (isDisposing)
            {
                if (_writer != null)
                {
                    _writer.Flush();
                    _writer.Dispose();
                }
            }

            _isDisposed = true;
        }

        ~HandwrittenDigitRecogniser()
        {
            Dispose(false);
        }
    }
}