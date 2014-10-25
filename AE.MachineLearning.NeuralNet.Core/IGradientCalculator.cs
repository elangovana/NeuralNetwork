using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Calcuates the gradient.
    /// </summary>
   public interface IGradientCalculator
   {
       /// <summary>
       /// Computes the gradient for the output layer
       /// </summary>
       /// <param name="target">The expected output value</param>
       /// <param name="actual">The acutal output value</param>
       /// <returns>Returns the gradient</returns>
       double CalculateGradientOutputLayer(double target, double actual);
   }
}
