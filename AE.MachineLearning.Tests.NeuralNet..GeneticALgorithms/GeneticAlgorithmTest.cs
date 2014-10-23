using System;
using AE.MachineLearning.NeuralNet.Core;
using AE.MachineLearning.NeuralNet.GeneticAlgorithms;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AE.MachineLearning.Tests.NeuralNet.GeneticAlgorithms
{
    [TestClass]
    public class GeneticAlgorithmTest
    {
        private FeedForwardLayerNeuralNetworkFactory _feedForwardLayerNeuralNetworkFactory;
        private RankBasedSelector _selector;
        private Mutator _mutator;

        [TestInitialize]
        public void TestInit()
        {
            _feedForwardLayerNeuralNetworkFactory = new FeedForwardLayerNeuralNetworkFactory();
            _feedForwardLayerNeuralNetworkFactory.Activation = new HyperTanActivation();

            _selector = new RankBasedSelector();
            _mutator = new Mutator(_feedForwardLayerNeuralNetworkFactory);
        }

        [TestMethod]
        public void ShouldReturnOptimumNetwork()
        {

             var trainInputs = new[]
                {
                    new[] {.889, -1, .85},
                    new[] {.77, -1.0, .99}
                };

            var trainOutputs = new[]
                {
                    new[] {.66, -1.8, .62},
                    new[] {.77, -1.0, .66}
                };

              var testInputs = new[]
                {
                    new[] {.889, -1, .85},
                    new[] {.77, -1.0, .99}
                };

            var testOutputs = new[]
                {
                    new[] {.66, -1.8, .62},
                    new[] {.77, -1.0, .66}
                };

            var sut = new GeneticAlgorithm(3, 3, 4, 10, new ClassficationFitnessCalculator(),
                                           new BackPropagationTraining(
                                               new EntropyLossGradientCalc(new HyperTanActivation())),
                                           _feedForwardLayerNeuralNetworkFactory, _selector, _mutator)
                {
                    NumberOfGenerations = 2,
                    SampleSize = 5

                };

            //Act
           var result=  sut.Optimise(trainInputs, trainOutputs, testInputs, testOutputs);

           
        }
    }
}
