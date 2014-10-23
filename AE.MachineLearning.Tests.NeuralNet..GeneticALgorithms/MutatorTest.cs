using System;
using System.Collections.Generic;
using AE.MachineLearning.NeuralNet.Core;
using AE.MachineLearning.NeuralNet.GeneticAlgorithms;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace AE.MachineLearning.Tests.NeuralNet.GeneticAlgorithms
{
    [TestClass]
    public class MutatorTest
    {
        private Mock<INetworkFactory> _mockFactory;
        private Mock<IActivation> _mockActivation;
        private const int MutationSize = 5;

        private const int MaxNodes = 30;
        private const int MinNodes = 10;

        [TestInitialize]
        public void TestInit()
        {
            _mockFactory = new Mock<INetworkFactory>();
            _mockActivation = new Mock<IActivation>();

       


        }

        [TestMethod]
        public void ShouldMutate()
        {
            var sut = new Mutator(new FeedForwardLayerNeuralNetworkFactory(),MinNodes,MaxNodes,MutationSize);
            var listNetworks = new List<AbstractNetwork>()
            {
                new NeuralNetwork(3, 1, 1, new[] {5}, _mockActivation.Object)
            };

            sut.Mutate(listNetworks, 1.0);


        }
    }
}
