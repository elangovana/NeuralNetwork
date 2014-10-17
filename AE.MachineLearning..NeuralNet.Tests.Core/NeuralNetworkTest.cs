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
    }
}