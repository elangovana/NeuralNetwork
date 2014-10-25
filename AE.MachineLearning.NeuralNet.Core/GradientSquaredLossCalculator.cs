namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Calculates gradient using a squared loss cost function.
    /// </summary>
   public class GradientSquaredLossCalculator : IGradientCalculator
    {
        private readonly IActivation _activation;
        private readonly SquaredCostFunction _costFunction;

        public GradientSquaredLossCalculator(IActivation activation)
        {
            _activation = activation;
            _costFunction = new SquaredCostFunction();
        }

       /// <summary>
       /// Computes gradient using formula derivateOfActivationFunction * derivative of squares loss cost function.
       /// </summary>
       /// <param name="target">Expected value</param>
       /// <param name="actual">Actual Value</param>
       /// <returns>Gradient value</returns>
        public double CalculateGradientOutputLayer(double target, double actual)
        {
            return _activation.CalculateDerivative(actual)*_costFunction.DerivativeCostWrtOutput(target, actual);
        }
    }
}