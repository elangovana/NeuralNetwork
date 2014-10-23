using System.IO;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
    public interface IGeneticAlgorithm
    {
        StreamWriter LogWriter { get; set; }

        int NumberOfGenerations { get; set; }
        

        AbstractNetwork Optimise(double[][] trainInputs, double[][] trainOutputs, double[][] testInputs,
                                 double[][] testOutputs);
    }
}