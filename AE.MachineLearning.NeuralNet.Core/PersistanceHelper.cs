using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Xml;

namespace AE.MachineLearning.NeuralNet.Core
{
    static class  PersistanceHelper
    {
        public static void Serlialse<T>(T ser,  string fileName)
        {           

            var t = typeof(T);

            var stringXml = new StringBuilder();
            using (XmlWriter xw = XmlWriter.Create(stringXml))
            {
                var serializer = new DataContractSerializer(ser.GetType());
                serializer.WriteObject(xw, ser);
            }
            File.WriteAllText(fileName, stringXml.ToString());
        }

        public static T Deseralise<T>(string xmlFilePath)
        {
            T result;
            using (var stringReader = new StringReader(File.ReadAllText(xmlFilePath)))
            {
                using (XmlReader xw = XmlReader.Create(stringReader))
                {
                    var serializer = new DataContractSerializer(typeof(T));
                    result = (T)serializer.ReadObject(xw);
                }
            }
            return result;
        }
    }
}
