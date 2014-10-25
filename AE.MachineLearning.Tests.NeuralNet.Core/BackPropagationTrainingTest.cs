using System;
using System.Linq;
using AE.MachineLearning.NeuralNet.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace AE.MachineLearning.Tests.NeuralNet.Core
{
    [TestClass]
    public class BackPropagationTrainingTest
    {
       
        private GradientSquaredLossCalculator _gradientSquaredLossCalculator;
        private NeuralNetwork _network;
        private int _oIndex;


        [TestInitialize]
        public void TestInit()
        {
            _network = new NeuralNetwork(3, 2, 1, new[] {4}, new SigmoidActivation(), new HyperTanActivation());
            var weightsHidden = new double[4][];
            weightsHidden[0] = new[] {0.1, 0.5, 0.9};
            weightsHidden[1] = new[] {0.2, 0.6, 1.0};
            weightsHidden[2] = new[] {0.3, 0.7, 1.1};
            weightsHidden[3] = new[] {0.4, 0.8, 1.2};
            _network.SetWeightsForLayer(1, weightsHidden, new[] {-2.0, -6.0, -1.0, -7.0});

            var weightsOutput = new double[2][];
            weightsOutput[0] = new[] {1.3, 1.5, 1.7, 1.9};
            weightsOutput[1] = new[] {1.4, 1.6, 1.8, 2.0};
            _network.SetWeightsForLayer(2, weightsOutput, new[] {-2.5, -5.0});

            _oIndex = 2;


            _gradientSquaredLossCalculator = new GradientSquaredLossCalculator(new HyperTanActivation());

            //  _errorCalculator = new Mock<IErrorCalculator>();
            //  ToDo Fix
        }

        [TestMethod]
        public void ShouldTrainWithNoMomentum()
        {
            //Arrange

            var sut = new BackPropagationTraining(_network, _gradientSquaredLossCalculator)
                {
                    LearningRate = .90,
                    Momentum = .0,
                    MaxError = 0.0,
                    MaxIteration = 1,
                    ErrorCalculator = new MockErrorCalc()
                };


            var inputs = new[]
                {
                    new[] {1.0, 2.0, 3.0}
                };


            var outputs = new[]
                {
                    new[] {-.85, .7500}
                };

            //Act
            sut.Train(inputs, outputs);

            //Assert
            Assert.AreEqual(-.8932, Math.Round(_network.NetworkLayers[_oIndex].Neurons[0].Output, 4));
            Assert.AreEqual(-.8006, Math.Round(_network.NetworkLayers[_oIndex].Neurons[1].Output, 4));
        }


        [TestMethod]
        public void ShouldPredict()
        {
            //Arrange

            var sut = new BackPropagationTraining(_network, _gradientSquaredLossCalculator)
            {
                LearningRate = .90,
                Momentum = .0,
                MaxError = 0.0,
                MaxIteration = 1,
                ErrorCalculator = new MockErrorCalc()
            };


            var inputs = new[]
                {
                    new[] {1.0, 2.0, 3.0}
                };


            var outputs = new[]
                {
                    new[] {-.85, .7500}
                };

            //Act
            sut.Train(inputs, outputs);

            //Assert
            sut.Predict(inputs);
        }


        [TestMethod]
        public void ShouldTrainWithMometum()
        {
            //Arrange
            var sut = new BackPropagationTraining(_network, _gradientSquaredLossCalculator)
                {
                    LearningRate = .90,
                    Momentum = 0.2,
                    MaxError = 0.0000,
                    MaxIteration = 4,
                    ErrorCalculator = new MockErrorCalc()
                };
            var inputs = new[]
                {
                    new[] {1.0, 2.0, 3.0}
                };


            var outputs = new[]
                {
                    new[] {-.85, .7500}
                };

            //Act
            sut.Train(inputs, outputs);

            //Assert
            Assert.AreEqual(-.8859, Math.Round(_network.NetworkLayers[_oIndex].Neurons[0].Output, 4));
            Assert.AreEqual(.8460, Math.Round(_network.NetworkLayers[_oIndex].Neurons[1].Output, 4));
        }

        private class MockErrorCalc : IErrorCalculator
        {
            public double CalculateError(double[][] targetOutputs, double[][] actutalOutputs)
            {
                return 100.0;
            }

           
        }
    }
}