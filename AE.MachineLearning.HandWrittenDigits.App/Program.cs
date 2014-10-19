using System;
using System.IO;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.HandWrittenDigits.App
{
    internal class Program
    {

        private static void Main(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Usage: AE.MachineLearning.HandWrittenDigits.App.exe trainFilePath testFilePath outDir [<networkfile>]  [learningRate] [momentum]");
                return;
            }
            string trainFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[0]);
            string testFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[1]);

            string outDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[2]);

            string networkFile = null;
            if (args.Length > 3)
                networkFile = args[3];

            using (var handwrittenDigitRecogniser = new HandwrittenDigitRecogniser(trainFile, testFile, outDir, networkFile))
            {
                handwrittenDigitRecogniser.Run();
            }
            
        }
    }
}