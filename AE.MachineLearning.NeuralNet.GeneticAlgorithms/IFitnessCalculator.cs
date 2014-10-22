using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AE.MachineLearning.NeuralNet.Core
{
    interface IFitnessCalculator
    {
        double Calculator(double[][] targetOutputs, double[][] actualOutputs);
    }
}
