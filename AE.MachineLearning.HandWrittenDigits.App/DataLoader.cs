using System.Collections.Generic;
using System.IO;

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
            for (int r = 0; r < dataEntries.Count; r++)
            {
                double[] entry = dataEntries[r];
                Outputs[r] = new double[10];
                Inputs[r] = new double[entry.Length - 1];
                Outputs[r][(int) entry[0]] = 1.0;
                for (var c = 1; c < entry.Length; c++)
                {
                    Inputs[r][c-1] = entry[c];
                }
            }

            TestInputs = ParseFile(testFile).ToArray();
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
                    data.Add( new double[columns.Length]);
                  
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