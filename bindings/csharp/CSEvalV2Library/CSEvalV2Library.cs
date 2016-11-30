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

        // Todo: VariableKind contains some more types than just input/output, which could cause some confusions.
        // We probably want to have a new enum just for input/output
        public IDictionary<string, IEnumerable<ulong>> GetNodesShape(VariableKind nodeKind)
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

        // Todo: VariableKind contains some more types than just input/output, which could cause some confusions.
        // We probably want to have a new enum just for input/output
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
        public Value CreateValue<T>(string varName, List<List<T>> sequences, DeviceDescriptor computeDevice)
        {
            var variable = getVariableByName(varName);
            var inputDim = variable.Shape().TotalSize();

            if (typeof(T).Equals(typeof(float)))
            {
                var inputSeqVector = new FloatVectorVector();
                foreach (var seq in sequences)
                {
                    if (seq.Count() % inputDim != 0)
                    {
                        throw new InvalidDataException("the number of data in sequences does not match the input dimension");
                    }
                    var samples = new FloatVector(seq);
                    inputSeqVector.Add(samples);
                }
                var inputValue = Value.CreateDenseFloat(variable.Shape(), inputSeqVector, computeDevice);
                return inputValue;
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                var inputSeqVector = new DoubleVectorVector();
                foreach (var seq in sequences)
                {
                    if (seq.Count() % inputDim != 0)
                    {
                        throw new InvalidDataException("the number of data in sequences does not match the input dimension");
                    }
                    var samples = new DoubleVector(seq);
                    inputSeqVector.Add(samples);
                }
                var inputValue = Value.CreateDenseDouble(variable.Shape(), inputSeqVector, computeDevice);
                return inputValue;
            }
            else
            {
                throw new InvalidDataException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
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
