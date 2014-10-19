using System;
using AE.MachineLearning.HandWrittenDigits.App;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AE.MachineLearning.HandWrittenDigits.Test.App
{
    [TestClass]
    [DeploymentItem(TrainFile)]
    [DeploymentItem(TestFile)]
    public class HandwrittenDigitRecogniserTest
    {

        public const string TrainFile = "train.csv";
        public const string TestFile = "test.csv";


        [TestMethod]
        public void ShouldRun()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);
            var outDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "outDir");

            var testFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TestFile); ;

            using (var sut = new HandwrittenDigitRecogniser(trainFile,testFile, outDir))
            {
                sut.Run();
            }
          
        }
    }
}
