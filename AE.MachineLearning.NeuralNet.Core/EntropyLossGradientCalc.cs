namespace AE.MachineLearning.NeuralNet.Core
{
   public class EntropyLossGradientCalc : IGradientCalculator
    {
        private readonly IActivation _activation;

        /// <summary>
        /// Note the activation function must be a sigmiod or hypertan activation for this to work!!!
        /// </summary>
        /// <param name="activation"></param>
        public EntropyLossGradientCalc(IActivation activation)
        {
            _activation = activation;
        }

        public double CalculateGradientOutputLayer(double target, double actual)
        {
            return target - actual;
        }
    }
}