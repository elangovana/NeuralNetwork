using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace AE.MachineLearning.HandWrittenDigits.App
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