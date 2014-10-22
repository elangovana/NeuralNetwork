using System;
using System.IO;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.HandWrittenDigitRecogniser
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

        public void Run(int maxIteration = 10000, double maxError = .05)
        {
            Writelog("Starting Run");

            var data = new HandandWrittenDataLoader();
            data.LoadData(_trainFile, _testFile);

            var netWork = CreateNetwork(_networkFile, data);


            //Write Network init
            BackPropagationTraining trainingAlgoritihmAlgorithm = Train(data, netWork, maxIteration, maxError);

            Predict(data, trainingAlgoritihmAlgorithm);

           

            Writelog("Procesing complete");
        }

        private void Predict(HandandWrittenDataLoader data, BackPropagationTraining trainingAlgoritihmAlgorithm)
        {
            Writelog(string.Format("Running prediction with test records rows {0} columns {1}", data.TestInputs.Length,
                                   data.TestInputs[0].Length));
            double[][] prediction = trainingAlgoritihmAlgorithm.Predict(data.TestInputs);

            data.WriteData(_testFile, prediction, Path.Combine(_outDir, "predictions.csv"));

            var percentageCorrect = 0.0;
            if (data.GetCorrectTestRate(prediction, out percentageCorrect))
            {
                Writelog(string.Format("Percentage correct prediction {0}", percentageCorrect.ToString("F4")));
            }
           
        }

        private BackPropagationTraining Train(HandandWrittenDataLoader data, AbstractNetwork netWork, int maxIteration, double maxError)
        {
           
            Writelog(string.Format("Train file Records rows {0} columns {1}", data.Inputs.Length, data.Inputs[0].Length));
            Writelog(string.Format("Begining training using learning rate {0}, momentum {1}, maxIteration {2}, maxError {3}", _learningRate, _momentum, maxIteration, maxError));
         
            var trainingAlgorithm = new BackPropagationTraining(netWork, new EntropyLossGradientCalc(new HyperTanActivation()) )
                {
                    LogWriter = RunLogWriter
                };
         
            trainingAlgorithm.Train(data.Inputs, data.Outputs,
                                    _learningRate, _momentum, maxError,maxIteration);

            netWork.PersistNetwork(Path.Combine(_outDir, "NetworkFinal.xml"));
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