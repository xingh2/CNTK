//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CSEvalV2Library -- C# Eval V2 Library
//

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CSharp
{
    public class Evaluation
    {
        public Evaluation()
        {
            model = null;
        }

        public void LoadModel(string modelFile, DeviceDescriptor computeDevice)
        {
            if (!File.Exists(modelFile))
            {
                throw new FileNotFoundException(string.Format("File '{0}' not found.", modelFile));
            }
            Function.LoadModel(modelFile, computeDevice);
        }

        // Todo: set default parameters  = DeviceDescriptor.UseDefaultDevice()
        public void Evaluate(Dictionary<Variable, Value> arguments, Dictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
        {
            if (model == null)
            {
                throw new NullReferenceException("No model is loaded. Please load the model first before evaluation.");
            }

            // Evaluate the model.

        }

        public void Clone()
        {

        }

        private Function model;

    }
}
