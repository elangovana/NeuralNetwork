namespace AE.MachineLearning.NeuralNet.Core
{
    /// <summary>
    /// Randomises the order of the data set 
    /// </summary>
    public interface IShuffler
    {

        /// <summary>
        /// Randomises the order of the input data sets and updates the order of the outputs accordingly.
        /// </summary>
        /// <param name="inputs">The input data to randomise</param>
        /// <param name="outputs">The ouput of the corresponding input data</param>
        /// <param name="randomisedInputs">Randomised input data</param>
        /// <param name="randomisedOutputs">The output data corresponding to the result of randomised input</param>
        void Shuffle(double[][] inputs, double[][] outputs, out double[][] randomisedInputs,
                      out double[][] randomisedOutputs);
    }
}