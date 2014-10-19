using System;
using AE.MachineLearning.HandWrittenDigits.App;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AE.MachineLearning.HandWrittenDigits.Test.App
{
    [TestClass]
    [DeploymentItem(TrainFile)]
    public class HandwrittenDigitRecogniserTest
    {

        public const string TrainFile = "train.csv";


        [TestMethod]
        public void ShouldRun()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);
            var outDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "outDir");

            var testFile = trainFile;

            using (var sut = new HandwrittenDigitRecogniser(trainFile,testFile, outDir))
            {
                sut.Run();
            }
          
        }
    }
}
