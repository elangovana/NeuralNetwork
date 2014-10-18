using System;
using System.IO;
using AE.MachineLearning.NeuralNet.Core;

namespace AE.MachineLearning.HandWrittenDigits.App
{
    internal class Program
    {
        private readonly HandwrittenDigitRecogniser _handwrittenDigitRecogniser = new HandwrittenDigitRecogniser();

        private static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("Usage: AE.MachineLearning.HandWrittenDigits.App.exe <trainFilePath> <testFilePath>");
                return;
            }
            string trainFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[0]);
            string testFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, args[1]);

            HandwrittenDigitRecogniser.Calculate(trainFile, testFile);
        }
    }
}