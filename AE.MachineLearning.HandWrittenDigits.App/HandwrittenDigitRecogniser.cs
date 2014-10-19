using System;
using System.IO;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.HandWrittenDigits.App
{
    public class HandwrittenDigitRecogniser : IDisposable
    {
        private readonly string _networkFile;
        private readonly string _outDir;
        private readonly string _testFile;
        private readonly string _trainFile;
        private bool _isDisposed;
        private readonly double _learningRate;
        private readonly double _momentum;
        private StreamWriter _writer;

        public HandwrittenDigitRecogniser(string trainFile, string testFile, string outDir, double learningRate, double momentum,   string networkFile = null)
        {
            _trainFile = trainFile;
            _testFile = testFile;
            _outDir = Path.Combine(outDir, string.Format("Run{0}", DateTime.Now.ToString("yyyyMMddmmhhss")));
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

        public void Run(int maxIteration = 10000)
        {
            Writelog("Starting Run");

            var data = new HandandWrittenDataLoader();
            data.LoadData(_trainFile, _testFile);

            NeuralNetwork netWork = CreateNetwork(_networkFile, data);


            //Write Network init
            BackPropagationTraining trainingAlgorithm = Train(data, netWork, maxIteration);

            Predict(data, trainingAlgorithm);

            Writelog("Procesing complete");
        }

        private void Predict(HandandWrittenDataLoader data, BackPropagationTraining trainingAlgorithm)
        {
            Writelog(string.Format("Running prediction with test records rows {0} columns {1}", data.TestInputs.Length,
                                   data.TestInputs[0].Length));
            double[][] prediction = trainingAlgorithm.Predict(data.TestInputs);

            data.WriteData(_testFile, prediction, Path.Combine(_outDir, "predictions.csv"));
        }

        private BackPropagationTraining Train(HandandWrittenDataLoader data, NeuralNetwork netWork, int maxIteration)
        {
           
            Writelog(string.Format("Train file Records rows {0} columns {1}", data.Inputs.Length, data.Inputs[0].Length));
            Writelog(string.Format("Begining training using learning rate {0}, momentum {1}", _learningRate, _momentum));
         
            var trainingAlgorithm = new BackPropagationTraining(netWork, new SquaredCostFunction())
                {
                    LogWriter = RunLogWriter
                };
         
            trainingAlgorithm.Train(data.Inputs, data.Outputs,
                                    _learningRate, _momentum,.05,maxIteration);
            return trainingAlgorithm;
        }


        public void Writelog(string message)
        {
            RunLogWriter.WriteLine("{0} - {1}", DateTime.Now, message);
        }

        private NeuralNetwork CreateNetwork(string networkFile, HandandWrittenDataLoader data)
        {
            NeuralNetwork netWork;
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

            netWork.PersistNetwork(Path.Combine(_outDir, "NetworkOut.xml"));
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