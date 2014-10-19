using System;
using System.IO;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.HandWrittenDigits.App
{
    internal class Program
    {

        private static void Main(string[] args)
        {
            if (args.Length < 6)
            {
                Console.WriteLine("Usage: AE.MachineLearning.HandWrittenDigits.App.exe trainFilePath testFilePath outDir learningRate momentum maxIteration [<networkfile>]  ");
                return;
            }
            string trainFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[0]);
            string testFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[1]);

            string outDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[2]);

            double learningRate = Double.Parse(args[3]);
            double mometum = Double.Parse(args[4]);
            int maxIteration = int.Parse(args[5]);
            string networkFile = null;
            if (args.Length == 7)
                networkFile = args[6];

            using (var handwrittenDigitRecogniser = new HandwrittenDigitRecogniser(trainFile, testFile, outDir, learningRate,mometum, networkFile))
            {
                handwrittenDigitRecogniser.Run(maxIteration);
            }
            
        }
    }
}