namespace AE.MachineLearning.NeuralNet.Core
{
   public class GradientSquaredLossCalculator : IGradientCalculator
    {
        private readonly IActivation _activation;
        private readonly SquaredCostFunction _costFunction;

        public GradientSquaredLossCalculator(IActivation activation)
        {
            _activation = activation;
            _costFunction = new SquaredCostFunction();
        }

        public double CalculateGradientOutputLayer(double target, double actual)
        {
            return _activation.CalculateDerivative(actual)*_costFunction.DerivativeCostWrtOutput(target, actual);
        }
    }
}