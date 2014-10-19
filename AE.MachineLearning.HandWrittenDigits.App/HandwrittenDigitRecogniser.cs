﻿using System;
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
        private double _learningRate;
        private double _momentum;
        private StreamWriter _writer;

        public HandwrittenDigitRecogniser(string trainFile, string testFile, string outDir, string networkFile = null)
        {
            _trainFile = trainFile;
            _testFile = testFile;
            _outDir = Path.Combine(outDir, string.Format("Run{0}", DateTime.Now.ToString("yyyyMMddmmhhss")));
            PersistanceHelper.SetUpDir(_outDir);
            _networkFile = networkFile;
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

        public void Run()
        {
            
            Writelog("Starting Run");

            var data = new HandandWrittenDataLoader();
            data.LoadData(_trainFile, _testFile);

            NeuralNetwork netWork = CreateNetwork(_networkFile, data);

           

            //Write Network init


            _learningRate = .9;
            _momentum = 0.0;
            Writelog(string.Format("Begining training using learning rate {0}, momentum {1}", _learningRate, _momentum));
            new BackPropagationTraining(netWork, new SquaredCostFunction()).Train(data.Inputs, data.Outputs,
                                                                                  _learningRate, _momentum);
        }


        public void Writelog(string message)
        {
            RunLogWriter.WriteLine("{0} - {1}", DateTime.Now, message);
        }

        private static NeuralNetwork CreateNetwork(string networkFile, HandandWrittenDataLoader data)
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
                netWork = PersistanceHelper.Deseralise<NeuralNetwork>(networkFile);
            }
            return netWork;
        }


        private void Dispose(bool isDisposing)
        {
            if (_isDisposed) return;

            if (isDisposing)
            {
                if (_writer != null) _writer.Dispose();
            }

            _isDisposed = true;
        }

        ~HandwrittenDigitRecogniser()
        {
            Dispose(false);
        }
    }
}