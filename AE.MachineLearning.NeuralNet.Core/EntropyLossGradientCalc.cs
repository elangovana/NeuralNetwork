namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Computes the Gradient for  the last layer, as a product of derivative of the activation funtion and the cost function.
    /// </summary>
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

       /// <summary>
       /// Using short cut calc , assuming the activation layer is either a sigmiod or a tan function
       /// </summary>
       /// <param name="target">Expected value</param>
       /// <param name="actual">Actual Value</param>
       /// <returns>Entropy loss gradient</returns>
        public double CalculateGradientOutputLayer(double target, double actual)
        {
            return target - actual;
        }
    }
}