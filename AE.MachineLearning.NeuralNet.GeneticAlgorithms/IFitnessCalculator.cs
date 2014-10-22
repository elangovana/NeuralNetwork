namespace AE.MachineLearning.NeuralNet.GeneticAlgorithms
{
   public interface IFitnessCalculator
    {
        double Calculator(double[][] targetOutputs, double[][] actualOutputs);
    }
}
