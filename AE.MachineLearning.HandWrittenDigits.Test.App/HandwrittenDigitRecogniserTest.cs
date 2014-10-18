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
        public void ShouldCalculate()
        {
            var trainFile = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TrainFile);

            var testFile = trainFile;

            HandwrittenDigitRecogniser.Calculate( trainFile, testFile);
        }
    }
}
