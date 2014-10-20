using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AE.MachineLearning.NeuralNet.Core
{
   public interface IGradientCalculator
   {
       double CalculateGradientOutputLayer(double target, double actual);
   }
}
