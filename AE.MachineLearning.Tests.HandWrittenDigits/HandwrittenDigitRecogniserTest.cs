using System;
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
                sut.Run();
            }
          
        }

        [TestMethod]
        [DeploymentItem(GaTrainFile)]
        public void ShouldRunGeneticAlgorithm()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);
            var outDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "outDir");

            var testFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TestFile); ;

            using (var sut = new HandwrittenDigitRecogniser(GaTrainFile, testFile, outDir, .2, .5))
            {
                sut.RunGeneticAlgorithm(1,3,1,120,10, 10);
            }

            throw new Exception("testhkjhk");

        }


        [TestMethod]
        public void ShouldLoadFromFile()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);
            var outDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "outDir");

            var testFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TestFile); ;

            using (var sut = new HandwrittenDigitRecogniser(trainFile, testFile, outDir,.5,.2, NetworkFile))
            {
                sut.Run();
            }

        }
    }
}
