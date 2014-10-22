using System.IO;

namespace AE.MachineLearning.NeuralNet.Core
{
    internal interface IGeneticAlgorithm
    {
        StreamWriter LogWriter { get; set; }

        AbstractNetwork Optimise(double[][] trainInputs, double[][] trainOutputs, double[][] testInputs,
                                 double[][] testOutputs);
    }
}