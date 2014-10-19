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
    public class PersistanceHelper
    {
        public static void SetUpDir(string outDir)
        {
            if (!Directory.Exists(outDir))
                Directory.CreateDirectory(outDir);
        }

        public static void Serlialse<T>(T ser, string outDir, string fileName)
        {
            SetUpDir(outDir);

            var t = typeof (T);
          
            var stringXml = new StringBuilder();
            using (XmlWriter xw = XmlWriter.Create(stringXml))
            {
                var serializer = new XmlSerializer(ser.GetType());
                serializer.Serialize(xw, ser);
            }
            File.WriteAllText(Path.Combine(outDir, fileName), stringXml.ToString());
        }

        public static T Deseralise<T>(string xmlFilePath)
        {
            T result;
            using (var stringReader = new StringReader(File.ReadAllText(xmlFilePath)))
            {
                using (XmlReader xw = XmlReader.Create(stringReader))
                {
                    var serializer = new XmlSerializer(typeof(T));
                    result = (T) serializer.Deserialize(xw);
                }
            }
            return result;
        }
    }
}