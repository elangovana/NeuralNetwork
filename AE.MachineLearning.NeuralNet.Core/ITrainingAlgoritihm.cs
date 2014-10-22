using System.IO;

namespace AE.MachineLearning.NeuralNet.Core
{
    public interface ITrainingAlgoritihm
    {
        StreamWriter LogWriter { get; set; }

        /// <summary>
        ///     Trains the network
        /// </summary>
        /// <param name="inputs">The first dimenension is the dataset, the second dimension must be equal  number of input features </param>
        /// <param name="targetOutputs"></param>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        /// <param name="maxError">Max Error Value. Stops when the error rate reaches this value</param>
        /// <param name="maxIteration">To prevent long running times, stops at this iteration</param>
        void Train(double[][] inputs, double[][] targetOutputs, double learningRate, double momentum,
                   double maxError = .05, int maxIteration = 10000);

        /// <summary>
        ///     Predicts output based on the training
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        double[][] Predict(double[][] inputs);
    }
}