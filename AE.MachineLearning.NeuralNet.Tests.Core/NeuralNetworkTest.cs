using System;
using AE.MachineLearning.NeuralNet.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace AE.MachineLearning.NeuralNet.Tests.Core
{
    [TestClass]
    public class NeuralNetworkTest
    {
        private Mock<IActivation> _mockActivate;

        [TestInitialize]
        public void Init()
        {
            _mockActivate = new Mock<IActivation>();
        }

        [TestMethod]
        public void ShouldConstructNetwork()
        {
            //Arrange
            const int numOfInputs = 3;
            const int numOfOutputs = 2;
            const int numberOfHiddenLayers = 1;
            var neuronsForEachHidden = new[] {4};

            //Act
            var sut = new NeuralNetwork(numOfInputs, numOfOutputs, numberOfHiddenLayers, neuronsForEachHidden,
                                        _mockActivate.Object);

            //Assert
            Assert.AreEqual(numOfInputs, sut.NumberOfInputFeatures);
            Assert.AreEqual(numOfOutputs, sut.NumberOfOutputs);
            Assert.AreEqual(numberOfHiddenLayers, sut.NumberOfHiddenLayers);
            Assert.AreEqual(numberOfHiddenLayers + 2, sut.NetworkLayers.Length);

            //Assert
            Assert.AreEqual(3, sut.NetworkLayers[0].NumOfNeurons);
            Assert.AreEqual(4, sut.NetworkLayers[1].NumOfNeurons);
            Assert.AreEqual(2, sut.NetworkLayers[2].NumOfNeurons);
        }

        [TestMethod]
        public void ShouldSetWeights()
        {
            //Arrange
            const int numOfInputs = 3;
            const int numOfOutputs = 2;
            const int numberOfHiddenLayers = 1;
            var neuronsForEachHidden = new[] {4};
            var sut = new NeuralNetwork(numOfInputs, numOfOutputs, numberOfHiddenLayers, neuronsForEachHidden,
                                        _mockActivate.Object);
            var weights = new[]
                {
                    new[] {0.1},
                    new[] {0.2},
                    new[] {0.3}
                };

            //Act


            var biases = new[]
                {
                    0.4, 0.5, 0.6
                };

            sut.SetWeightsForLayer(0, weights, biases);

            //Assert 
            for (int i = 0; i < numOfInputs; i++)
            {
                Assert.AreEqual(weights[i][0], sut.NetworkLayers[0].Neurons[i].Weights[0]);
            }
            for (int i = 0; i < numOfInputs; i++)
            {
                Assert.AreEqual(biases[i], sut.NetworkLayers[0].Neurons[i].Bias);
            }
        }

        [TestMethod]
        public void ShouldComputeOutputCorrectly()
        {
             var sut = new NeuralNetwork(3, 2, 1, new[] {4}, new SigmoidActivation(),new HyperTanActivation());
            var weightsHidden = new double[4][];
            weightsHidden[0] = new[] {0.1, 0.5, 0.9};
            weightsHidden[1] = new[] {0.2, 0.6, 1.0};
            weightsHidden[2] = new[] {0.3, 0.7, 1.1};
            weightsHidden[3] = new[] {0.4, 0.8, 1.2};
            sut.SetWeightsForLayer(1, weightsHidden, new[] {-2.0, -6.0, -1.0, -7.0});

            var weightsOutput = new double[2][];
            weightsOutput[0] = new[] {1.3, 1.5, 1.7, 1.9};
            weightsOutput[1] = new[] {1.4, 1.6, 1.8, 2.0};
            sut.SetWeightsForLayer(2, weightsOutput, new[] {-2.5, -5.0});




            var inputs = new[] {1.0, 2.0, 3.0};

            
          

            //Act
            sut.ComputeOutput(inputs);

            //Assert
            Assert.AreEqual(0.7225, Math.Round(  sut.NetworkLayers[2].Neurons[0].Output,4));
            Assert.AreEqual(-0.8779,Math.Round(sut.NetworkLayers[2].Neurons[1].Output,4));
        }

        [TestMethod]
        public void ShouldSetRandomWeights()
        {
            var sut = new NeuralNetwork(3, 2, 1, new[] {4}, _mockActivate.Object);

            sut.InitNetworkWithRandomWeights();

        }
    }
}