using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AE.MachineLearning.HandWrittenDigitRecogniser
{
    public class HandandWrittenDataLoader
    {
        private const char Separator = ',';

        public double[][] Inputs { get; private set; }

        public double[][] TestInputs { get; private set; }

        public double[][] Outputs { get; private set; }

        private int[] TestOutputs { get; set; }

        public void LoadData(string trainFile, string testFile)
        {
            List<double[]> dataEntries = ParseFile(trainFile);
            //Set IO for train
            double[][] inputs;
            double[][] outputs;
            EncodeOutput(dataEntries, out inputs, out outputs);

            Inputs = inputs;
            Outputs = outputs;

            ProcessTestInputs(testFile, Inputs[0].Length);
        }

        public bool GetCorrectTestRate(double[][] outputs, out double percentageRate)
        {
            percentageRate = 0.0;
            if (TestOutputs == null) return false;

            int totalCorrect = outputs.Where((t, r) => TestOutputs[r] == GetDigit(t)).Count();

            percentageRate = (totalCorrect/(double) outputs.Length)*100.0;
            return true;
        }

        private void ProcessTestInputs(string testFile, int inputLength)
        {
            List<double[]> testInputs = ParseFile(testFile);

            TestInputs = testInputs.ToArray();
            if (TestInputs[0].Length == inputLength) return;

            //The test data doenst contain any outputs

            TestOutputs = new int[testInputs.Count];

            for (int r = 0; r < testInputs.Count; r++)
            {
                TestOutputs[r] = (int) testInputs[r][0];
            }


            double[][] parsedInput;
            double[][] parsedOutput;
            EncodeOutput(testInputs, out parsedInput, out parsedOutput);

            TestInputs = parsedInput;
        }

        private void EncodeOutput(List<double[]> dataEntries, out double[][] inputs, out double[][] outputs)
        {
            const int outputClasses = 10;
            outputs = new double[dataEntries.Count][];
            inputs = new double[dataEntries.Count][];
            for (int r = 0; r < dataEntries.Count; r++)
            {
                double[] entry = dataEntries[r];
                outputs[r] = new double[outputClasses];
                inputs[r] = new double[entry.Length - 1];
                //Init all classes to -1
                for (int o = 0; o < outputClasses; o++)
                {
                    outputs[r][o] = -1.0;
                }
                outputs[r][(int) entry[0]] = 1.0;
                for (int c = 1; c < entry.Length; c++)
                {
                    inputs[r][c - 1] = entry[c];
                }
            }
        }


        public void WriteData(string inputTestFile, double[][] outputs, string outFile)
        {
            using (StreamWriter writer = File.CreateText(outFile))
            {
                string[] input = File.ReadAllLines(inputTestFile);

                for (int i = 0; i < input.Length; i++)
                {
                    int digit = GetDigit(outputs[i]);

                    writer.WriteLine("{0},{1}", digit, input[i]);
                }
            }
        }

        private static int GetDigit(double[] outputs)
        {
            double maxProb = outputs.Max(x => x);

            int digit = -1;
            for (int j = 0; j < outputs.Length; j++)
            {
                if (Math.Round(outputs[j], 4) == Math.Round(maxProb, 4))
                {
                    digit = j;
                    break;
                }
            }
            return digit;
        }

        private static List<double[]> ParseFile(string dataFile)
        {
            var data = new List<double[]>();
            using (StreamReader file = File.OpenText(dataFile))
            {
                string line;

                int i = 0;
                while ((line = file.ReadLine()) != null)
                {
                    string[] columns = line.Split(new[] {Separator});
                    data.Add(new double[columns.Length]);

                    for (int c = 0; c < columns.Length; c++)
                    {
                        data[i][c] = double.Parse(columns[c]);
                    }
                    i++;
                }
            }

            return data;
        }

        public void Randomise( double[][] inputs,  double[][]outputs, out double[][]  randomisedInputs, out double[][]  randomisedOutputs)
        {
            var random = new Random();
            var randomOrder = new double[inputs.Length];
            int randomLimit = inputs.Length * 2;
            for (int i = 0; i < randomOrder.Length; i++)
            {
                randomOrder[i] = random.Next(0, randomLimit);
            }

            var randomised = inputs.Select((r, i) => new {input = r, output = outputs[i], order = randomOrder[i]})
                                   .OrderBy(x => x.order).ToArray();

            randomisedInputs = new double[inputs.Length][];
            randomisedOutputs = new double[outputs.Length][];
            for (int i = 0; i < randomised.Count(); i++)
            {
                randomisedInputs[i] = randomised[i].input;
                randomisedOutputs[i] = randomised[i].output;
            }
        }
    }
}