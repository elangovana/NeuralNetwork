using System.IO;

namespace AE.MachineLearning.NeuralNet.Core
{
    public interface ITrainingAlgoritihm
    {
        StreamWriter LogWriter { get; set; }

        AbstractNetwork Network { get; set; }

        /// <summary>
        ///     Trains the network
        /// </summary>
        /// <param name="inputs">The first dimenension is the dataset, the second dimension must be equal  number of input features </param>
        /// <param name="targetOutputs"></param>
        
        void Train(double[][] inputs, double[][] targetOutputs);

        /// <summary>
        ///     Predicts output based on the training
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        double[][] Predict(double[][] inputs);
    }
}