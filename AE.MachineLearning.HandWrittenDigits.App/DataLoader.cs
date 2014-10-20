using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AE.MachineLearning.HandWrittenDigits.App
{
    public class HandandWrittenDataLoader
    {
        private const char Separator = ',';

        public double[][] Inputs { get; private set; }

        public double[][] TestInputs { get; private set; }

        public double[][] Outputs { get; private set; }


        public void LoadData(string trainFile, string testFile)
        {
            List<double[]> dataEntries = ParseFile(trainFile);
            //Set IO for train
            Inputs = new double[dataEntries.Count][];
            Outputs = new double[dataEntries.Count][];
            EncodeOutput(dataEntries);

            TestInputs = ParseFile(testFile).ToArray();
        }

        private void EncodeOutput(List<double[]> dataEntries)
        {
            const int outputClasses = 10;

            for (int r = 0; r < dataEntries.Count; r++)
            {
                double[] entry = dataEntries[r];
                Outputs[r] = new double[outputClasses];
                Inputs[r] = new double[entry.Length - 1];
                //Init all classes to -1
                for (int o = 0; o < outputClasses; o++)
                {
                    Outputs[r][o] = -1.0;
                }
                Outputs[r][(int)entry[0]] = 1.0;
                for (int c = 1; c < entry.Length; c++)
                {
                    Inputs[r][c - 1] = entry[c];
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
                    double maxProb = outputs[i].Max(x => x);

                    int digit = -1;
                    for (int j = 0; j < outputs[i].Length; j++)
                    {
                        if (Math.Round(outputs[i][j], 4) == Math.Round(maxProb, 4))
                        {
                            digit = j;
                            break;
                        }
                    }

                    writer.WriteLine("{0},{1}", digit, input[i]);
                }
            }
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
    }
}