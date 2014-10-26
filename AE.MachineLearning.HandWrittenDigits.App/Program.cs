using System;
using System.IO;
using AE.MachineLearning.HandWrittenDigitRecogniser;

namespace AE.MachineLearning.HandWrittenDigits.App
{
    internal class Program
    {
        private static string OptionBP;
        private static string OptionGA;

        private static void Main(string[] args)
        {
            OptionBP = "BP";
            OptionGA = "GA";
            if (args.Length < 1 || (args.Length > 1 && (args[0] != OptionBP && args[0] != OptionGA)))
            {
                Console.WriteLine("Run from command prompt");
                PrintUsage();
                Console.WriteLine("\nPress enter to exit....");
                Console.ReadLine();
                return;
            }
            var parsedArgs = new string[args.Length - 1];
            for (int i = 1, j = 0; i < args.Length; i++ ,j++)
            {
                parsedArgs[j] = args[i];
            }
            if (args[0] == OptionBP)
                RunBackProp(parsedArgs);
            else if (args[0] == OptionGA)
            {
                RunGA(parsedArgs);
            }
            else
            {
                PrintUsage();
                return;
            }
        }

        private static void RunGA(string[] args)
        {

            if (args.Length < 14)
            {
                Console.WriteLine("Incorrect number of arguments supplied to run Ga");
                PrintUsage();
                return;
            }
            string trainFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[0]);
            string testFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[1]);

            string outDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[2]);

            int minLayer = int.Parse(args[3]);
            int maxLayer = int.Parse(args[4]);
            int minNodesPerLayer = int.Parse(args[5]);
            int maxNodesPerLayer = int.Parse(args[6]);
            int mutataionSize = int.Parse(args[7]);
            int populationSize = int.Parse(args[8]);
            int numGenerations = int.Parse(args[9]);
            double learningRate = Double.Parse(args[10]);
            double mometum = Double.Parse(args[11]);
            int maxIteration = int.Parse(args[12]);
            double maxError = Double.Parse(args[13]);


            using (
                var handwrittenDigitRecogniser = new HandwrittenDigitRecogniser(trainFile, testFile, outDir,
                                                                                learningRate,
                                                                                mometum))
            {
                Console.WriteLine("Running genetic algorithm with the parameters supplied. \n Please check the outdir specified for logs and output...");
                handwrittenDigitRecogniser.RunGeneticAlgorithm(minLayer, maxLayer, minNodesPerLayer, maxNodesPerLayer,
                                                               numGenerations, populationSize, 1, mutataionSize,
                                                               maxIteration, maxError);
                Console.WriteLine("Completed.. Please check output directory for results");
            }
        }

        private static void PrintUsage()
        {
            
            Console.WriteLine("Usage: " +
                              "\n--------------------------------" +
                              "\n To Run Back propogation" +
                              "\n   AE.MachineLearning.HandWrittenDigits.App.exe BP <trainFilePat> <testFilePath> <outDir> <learningRate> <momentum> <maxIteration> <maxError> [<networkfile>]  " +
                              "\n                NetworkFile is optional" +
                              "\n" +
                              "\n" +
                              "\n To Run Genetic Algorithm" +
                              "\n   AE.MachineLearning.HandWrittenDigits.App.exe GA <trainFilePat> <testFilePath> <outDir> <minLayer> <maxLayer> <minNodesPerLayer> <maxNodesPerLayer>  <mutationSize> <populationSize> <NoOfGenerations> <learningRate> <momentum> <maxIteration> <maxError>");
        }


        private static void RunBackProp(string[] args)
        {
            if (args.Length < 8)
            {
                Console.WriteLine("Incorrect parameters supplied!!");
                PrintUsage();
                return;
            }

            string trainFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[0]);
            string testFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[1]);

            string outDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[2]);

            double learningRate = Double.Parse(args[3]);
            double mometum = Double.Parse(args[4]);
            int maxIteration = int.Parse(args[5]);
            double maxError = Double.Parse(args[6]);
            string networkFile = null;
            if (args.Length == 8)
                networkFile = args[7];

            using (
                var handwrittenDigitRecogniser = new HandwrittenDigitRecogniser(trainFile, testFile, outDir,
                                                                                learningRate,
                                                                                mometum, networkFile))
            {
                Console.WriteLine("Running Back propagation with the parameters supplied. \n Please check the outdir specified for logs and output...");
                handwrittenDigitRecogniser.Run(maxIteration, maxError);

                Console.WriteLine("Completed.. Please check output directory for results");
            }
        }
    }
}