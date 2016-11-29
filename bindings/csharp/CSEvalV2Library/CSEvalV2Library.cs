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
            rootFunction = null;
        }

        public void LoadModel(string rootFunctionFile, DeviceDescriptor computeDevice)
        {
            if (!File.Exists(rootFunctionFile))
            {
                throw new FileNotFoundException(string.Format("File '{0}' not found.", rootFunctionFile));
            }
            Function.LoadModel(rootFunctionFile, computeDevice);
        }

        public IDictionary<string, IEnumerable<ulong>> GetNodesLayout(VariableKind nodeKind)
        {
            var retVal = new Dictionary<string, IEnumerable<ulong>>();

            IEnumerable<Variable> varList;
            if (nodeKind == VariableKind.Input)
            {
                varList = rootFunction.Arguments();
            }
            else if (nodeKind == VariableKind.Output)
            {
                varList = rootFunction.Outputs();
            }
            else 
            {
                // Todo: Use nameof after VS2015.
                throw new ArgumentException("Node kind must be '" + "VariableKind.Input" + "' or '" + "VariableKind.Output" + "'.");
            }

            foreach (var arg in varList)
            {
                if (retVal.ContainsKey(arg.Name()))
                {
                    throw new Exception("duplicated name '" + arg.Name() + "'.");
                }
                var dim = new List<ulong>();
                // The Dimensions is IEnumerable<uint>
                // Todo: fix the swig to output IEnumberable<ulong>
                foreach (var d in arg.Shape().Dimensions())
                {
                    dim.Add(d);
                }
                retVal.Add(arg.Name(), dim);
            }

            return retVal;
        }

        public IDictionary<string, ulong> GetNodesSize(VariableKind nodeKind)
        {
            var retVal = new Dictionary<string, ulong>();

            IEnumerable<Variable> varList;
            if (nodeKind == VariableKind.Input)
            {
                varList = rootFunction.Arguments();
            }
            else if (nodeKind == VariableKind.Output)
            {
                varList = rootFunction.Outputs();
            }
            else 
            {
                // Todo: Use nameof after VS2015.
                throw new ArgumentException("Node kind must be '" + "VariableKind.Input" + "' or '" + "VariableKind.Output" + "'.");
            }

            foreach (var arg in varList)
            {
                if (retVal.ContainsKey(arg.Name()))
                {
                    throw new Exception("duplicated name '" + arg.Name() + "'.");
                }

                retVal.Add(arg.Name(), arg.Shape().TotalSize());
            }

            return retVal;
        }

        // Todo: set default parameters  = DeviceDescriptor.UseDefaultDevice()
        public void Evaluate(Dictionary<string, Value> arguments, Dictionary<string, Value> outputs, DeviceDescriptor computeDevice)
        {
            if (rootFunction == null)
            {
                throw new NullReferenceException("No rootFunction is loaded. Please load the rootFunction first before evaluation.");
            }

            // Evaluate the rootFunction.
            var argMap = new UnorderedMapVariableValuePtr();
            foreach (var p in arguments)
            {
                var variable = rootFunction.Arguments().Where(v => string.Equals(v.Name(), p.Key)).FirstOrDefault();
                if (variable == null)
                {
                    throw new KeyNotFoundException("No input variable '" + p.Key + "' found.");
                }
                argMap.Add(variable, p.Value);
            }

            var outMap = new UnorderedMapVariableValuePtr();
            foreach (var p in outputs)
            {
                var variable = rootFunction.Outputs().Where(v => string.Equals(v.Name(), p.Key)).FirstOrDefault();
                if (variable == null)
                {
                    throw new KeyNotFoundException("No output variable '" + p.Key + "' found.");
                }
                outMap.Add(variable, p.Value);
            }

            rootFunction.Evaluate(argMap, outMap, computeDevice);

            foreach (var p in outMap)
            {
                outputs[p.Key.Name()] = p.Value;
            }
        }

        // Create Value based on dense input
        // Todo: could this be a extension to Value class??
        public Value CreateValue<T>(string varName, List<List<T>> data)
        {
            var variable = getVariableByName(varName);
            var inputDim = variable.Shape().TotalSize();

            if (typeof(T).Equals(float))
            {
                var inputVector = new FloatVectorVector(inputData);
            var data = new FloatVectorVector() { inputVector };
            // Create value directly from data.
            var inputValue = Value.CreateDenseFloat(inputVar.Shape(), data, DeviceDescriptor.CPUDevice());
            }
            
        }

        public void Clone()
        {

        }

        private Function rootFunction;

        private Variable getVariableByName(string name)
        {
            var v = rootFunction.Arguments().Where(variable => string.Equals(variable.Name(), name)).FirstOrDefault();
            if (v == null)
            {
                v = rootFunction.Outputs().Where(variable => string.Equals(variable.Name(), name)).FirstOrDefault();
            }

            return v;
        }

    }
}
