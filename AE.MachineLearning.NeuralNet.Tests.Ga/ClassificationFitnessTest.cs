using System;
using AE.MachineLearning.NeuralNet.GeneticAlgorithms;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AE.MachineLearning.NeuralNet.Tests.GeneticAlgorithms
{
    [TestClass]
    public class ClassficationFitnessCalculatorTest
    {
        [TestMethod]
        public void ShouldCalculate()
        {
            //Arrange
            var targetOutputs = new[]
                {
                    new[] {.889, -1, .85},
                    new[] {.77, -1.0, .99}
                };

            var actual = new[]
                {
                    new[] {.66, -1.8, .62},
                    new[] {.77, -1.0, .66}
                };

            var sut = new ClassficationFitnessCalculator();

            //Act
            double result = sut.Calculator(targetOutputs, actual);

            //Assert
            Assert.AreEqual(50.0, Math.Round(result,1));
        }
    }
}