using System;
using AE.MachineLearning.HandWrittenDigitRecogniser;
using AE.MachineLearning.HandWrittenDigits;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AE.MachineLearning.HandWrittenDigits.Test.App
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
