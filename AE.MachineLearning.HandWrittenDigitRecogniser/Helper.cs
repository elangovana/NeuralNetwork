using System.IO;

namespace AE.MachineLearning.HandWrittenDigitRecogniser
{
    public class Helper
    {
        public static void SetUpDir(string outDir)
        {
            if (!Directory.Exists(outDir))
                Directory.CreateDirectory(outDir);
        }

       
    }
}