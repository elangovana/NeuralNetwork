﻿using System;
using AE.MachineLearning.HandWrittenDigitRecogniser;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AE.MachineLearning.Tests.HandWrittenDigits
{
    [TestClass]
    [DeploymentItem(TrainFile)]
    [DeploymentItem(TestFile)]
    [DeploymentItem(NetworkFile)]
    public class HandwrittenDigitRecogniserTest
    {

        public const string TrainFile = "train.csv";
        public const string TestFile = "test.csv";
        public const string NetworkFile = "network.xml";
        public const string GaTrainFile = "traindata.csv";

        [TestMethod]
        public void ShouldRun()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);
            var outDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "outDir");

            var testFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TestFile); ;

            using (var sut = new HandwrittenDigitRecogniser(trainFile,testFile, outDir,.9,.2))
            {
                sut.Run(2);
            }
          
        }

        [TestMethod]
        [DeploymentItem(GaTrainFile)]
        public void ShouldRunGeneticAlgorithm()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);
            var outDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "outDir");

            var testFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TestFile); ;

            using (var sut = new HandwrittenDigitRecogniser(GaTrainFile, testFile, outDir, .02, .7))
            {
                sut.RunGeneticAlgorithm(minLayers: 1, maxLayers: 2, maxNoOfNodes: 101, minNoOfNodes: 1,
                                        numberOfGenerations: 2, populationSize: 3, mutationSize: 5,
                                        iterationPerTraning: 1, maxIteration: 2, maxError: .29);
            }

           

        }


        [TestMethod]
        public void ShouldLoadFromFile()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);
            var outDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "outDir");

            var testFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TestFile); ;

            using (var sut = new HandwrittenDigitRecogniser(trainFile, testFile, outDir,.5,.2, NetworkFile))
            {
                sut.Run(2);
            }

        }
    }
}
