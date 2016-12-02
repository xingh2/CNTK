//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// program.cs -- Example for using C# Eval V2 API.
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSEvalV2Example
{
    public class Program
    {
        static void EvaluateUsingCSEvalLib()
        {
            // Load the model
            var model = new CNTK.CSharp.Evaluation();

            model.LoadModel("z.model", DeviceDescriptor.CPUDevice());

            // prepare input for evaluation
            int numOfSamples = 1;

            var inputDims = model.GetInputSizes();
            const string inputNodeName = "features";

            ulong numOfInputData = inputDims[inputNodeName];
            var inputData = new List<List<float>>(numOfSamples);
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[0].Add(i % 255);
            }

            var inputValue = model.CreateValue<float>(inputNodeName, inputData, DeviceDescriptor.CPUDevice());

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new Dictionary<string, Value>() { { inputNodeName, inputValue } };

            // Prepare output
            const string outputNodeName = "out.z_output";

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<string, Value>() { { outputNodeName, null } };

            // Evalaute
            // Todo: test on GPUDevice()?
            model.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice());

            var output = new List<List<float>>();
            var outputDims = model.GetOutputSizes();
            ulong numOfElementsInSample = outputDims[outputNodeName];

            model.CopyValueTo<float>(outputNodeName, outputMap[outputNodeName], output);

            ulong seqNo = 0;
            foreach (var seq in output)
            {
                var numOfSamplesInSequence = (ulong)seq.Count / numOfElementsInSample;
                ulong elementIndex = 0;
                ulong sampleIndex = 0;
                Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                foreach (var data in seq)
                {
                    if (elementIndex++ == 0)
                    {
                        Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                    }
                    Console.Write(" " + data);
                    if (elementIndex == numOfElementsInSample)
                    {
                        Console.WriteLine(".");
                        elementIndex = 0;
                        sampleIndex++;
                    }
                }
                seqNo++;
            }
        }

        //
        // The example shows how to prepare data based on dense input and get results as dense output.
        // The input data contains multiple sequences and each sequence contains multiple samples.
        //
        static void DenseExample()
        {
            // Load the model.
            Function myFunc = Function.LoadModel("resnet.model");

            // Get the input variable from by name
            const string inputNodeName = "features";
            // Todo: provide a help method in Function: getVariableByName()? Or has a property variables which is dictionary of <string, Variable>
            Variable inputVar = myFunc.Arguments().Where(variable => string.Equals(variable.Name(), inputNodeName)).FirstOrDefault();

            // Get shape data 
            NDShape inputShape = inputVar.Shape();
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize();

            // Number of sequences for this batch
            uint numOfSequences = 2;
            // Number of samples in each sequence
            uint[] numOfSamplesInSequence = { 4, 2 };

            // inputData contains all inputs for the evaluation
            // The inner List is the inputs for one sequence. Its size is inputShape.TotalSize() * numberOfSampelsInSequence
            // The outer List is the sequences. Its size is numOfSequences; 
            var inputData = new List<List<float>>();
            var fileList = new List<string>() { "zebra.jpg", "tiger.jpg", "deer.jpg", "pig.jpg", "buidling.jpg", "garden.jpg" };
            int fileIndex = 0;
            for (uint seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // Input data for the sequence
                var seqData = new List<float>();
                for (uint sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    var resized = bmp.Resize(imageWidth, imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();
                    // Add this sample to the list
                    seqData.AddRange(resizedCHW);
                }
                // Add this sequence to the list
                inputData.Add(seqData);
            }

            // Create value object from data.
            Value inputValue = Value.Create<float>(inputVar.Shape(), inputData, DeviceDescriptor.CPUDevice);

            // Create input map
            var inputMap = new Dictionary<Variable, Value>();
            inputMap.Add(inputVar, inputValue);

            // Repeat the steps above for each input.

            // Prepare output
            const string outputNodeName = "out.z_output";
            Variable outputVar = myFunc.Outputs().Where(variable => string.Equals(variable.Name(), outputNodeName)).FirstOrDefault();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<Variable, Value>();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // The buffer for storing output for this batch
            var outputData = new List<List<float>>();
            Value outputVal = outputMap[outputVar];
            // Get output result as dense output
            outputVal.CopyTo(outputData);

            // Output results
            var numOfElementsInSample = outputVar.Shape().TotalSize();
            ulong seqNo = 0;
            foreach (var seq in outputData)
            {
                ulong elementIndex = 0;
                ulong sampleIndex = 0;
                Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                foreach (var data in seq)
                {
                    // a new sample starts.
                    if (elementIndex++ == 0)
                    {
                        Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                    }
                    Console.Write(" " + data);
                    // reach the end of a sample.
                    if (elementIndex == numOfElementsInSample)
                    {
                        Console.WriteLine(".");
                        elementIndex = 0;
                        sampleIndex++;
                    }
                }
                seqNo++;
            }
        }

        // 
        // The example uses OneHot vector as input and output for evaluation
        // The input data contains multiple sequences and each sequence contains multiple samples.
        //
        static void OneHoteExample()
        {
            Function myFunc = Function.LoadModel("atis.model");
            var vocabToIndex = new Dictionary<string, long>();
            var indexToVocab = new Dictionary<long, string>();

            // Get input variable 
            const string inputNodeName = "features";
            Variable inputVar = myFunc.Arguments().Where(variable => string.Equals(variable.Name(), inputNodeName)).FirstOrDefault();
            // Todo: get size directly from inputVar.
            NDShape inputShape = inputVar.Shape();
            uint inputSize = inputShape.TotalSize();

            // The number of sequences in this batch
            int numOfSequences = 2;

            // The input data. 
            // Each sample is represented by a onehot vector, so the index of the non-zero value of each sample is saved in the inner list
            // The outer list represents sequences of the batch.
            var inputData = new List<List<long>>();
            var inputSentences = new List<string>() { 
                "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
                "BOS I want to book a flight from NewYork to Seattle EOS"
            };

            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // the input for one sequence 
                var seqData = new List<long>();
                // Get the word from the sentence.
                string[] substring = inputSentences[seqIndex].Split(' ');
                foreach (var str in substring)
                {
                    // Get the index of the word
                    var index = vocabToIndex[str];
                    // Add the sample to the sequence
                    seqData.Add(index);
                }
                // Add the sequence to the batch
                inputData.Add(seqData);
            }

            // Create the Value representing the data.
            Value inputValue = Value.Create<float>(inputVar.Shape(), inputData, DeviceDescriptor.CPUDevice);

            // Create input map
            var inputMap = new Dictionary<Variable, Value>();
            inputMap.Add(inputVar, inputValue);

            // Prepare output
            const string outputNodeName = "out.z_output";
            var outputVar = myFunc.Outputs().Where(variable => string.Equals(variable.Name(), outputNodeName)).FirstOrDefault();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<Variable, Value>();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            var outputData = new List<List<long>>();
            Value outputVal = outputMap[outputVar];

            // Get output as onehot vector
            outputVal.CopyTo(outputData);
            Debug.Assert(outputVar.Shape().Rank() == 1);
            var numOfElementsInSample = outputVar.Shape().TotalSize();

            // output the result
            ulong seqNo = 0;
            foreach (var seq in outputData)
            {
                Console.Write("Seq=" + seqNo + ":");
                foreach (var index in seq)
                {
                    // get the word based on index
                    Console.Write(indexToVocab[index]);
                }
                Console.WriteLine();
                // next sequence.
                seqNo++;
            }
        }

        // 
        // The example uses OneHot vector as input and output for evaluation
        // The input data contains multiple sequences and each sequence contains multiple samples.
        //
        static void SparseExample()
        {
            // Load the model.
            Function myFunc = Function.LoadModel("resnet.model");

            // Get the input variable from by name
            const string inputNodeName = "features";
            // Todo: provide a help method in Function: getVariableByName()? Or has a property variables which is dictionary of <string, Variable>
            Variable inputVar = myFunc.Arguments().Where(variable => string.Equals(variable.Name(), inputNodeName)).FirstOrDefault();

            // Get shape data 
            NDShape inputShape = inputVar.Shape();
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize();

            // Number of sequences for this batch
            uint numOfSequences = 2;
            // Number of samples in each sequence
            uint[] numOfSamplesInSequence = { 4, 2 };

            // inputData contains all inputs for the evaluation
            // The inner List is the inputs for one sequence. Its size is inputShape.TotalSize() * numberOfSampelsInSequence
            // The outer List is the sequences. Its size is numOfSequences; 
            var dataOfSequences = new List<List<float>>();
            var indexOfSequences = new List<List<long>>();
            var nnzCountOfSequences = new List<List<long>>();
            var fileList = new List<string>() { "zebra.jpg", "tiger.jpg", "deer.jpg", "pig.jpg", "buidling.jpg", "garden.jpg" };
            int fileIndex = 0;
            for (uint seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // Input data for the sequence
                var dataList = new List<float>();
                var indexList = new List<long>();
                var nnzCountList = new List<long>();
                for (uint sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    var resized = bmp.Resize(imageWidth, imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();
                    long nnzCount = 0;
                    long index = 0;
                    foreach (var v in resizedCHW)
                    {
                        // Put non-zero value into data
                        // put the index of this value into indexList
                        if (v != 0)
                        {
                            dataList.Add(v);
                            indexList.Add(index);
                            nnzCount++;
                        }
                        index++;
                    }
                    // Add nnzCount of this sample to nnzCountList
                    nnzCountList.Add(nnzCount);
                }
                // Add this sequence to the list
                dataOfSequences.Add(dataList);
                indexOfSequences.Add(indexList);
                nnzCountOfSequences.Add(nnzCountList);
            }

            // Create value object from data.
            Value inputValue = Value.Create<float>(inputVar.Shape(), dataOfSequences, indexOfSequences, nnzCountOfSequences, DeviceDescriptor.CPUDevice);

            // Create input map
            var inputMap = new Dictionary<Variable, Value>();
            inputMap.Add(inputVar, inputValue);

            // Repeat the steps above for each input.

            // Prepare output
            const string outputNodeName = "out.z_output";
            Variable outputVar = myFunc.Outputs().Where(variable => string.Equals(variable.Name(), outputNodeName)).FirstOrDefault();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<Variable, Value>();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // The buffer for storing output for this batch
            var outputData = new List<List<float>>();
            var outputIndex = new List<List<long>>();
            var outputNnzCount = new List<List<long>>();

            Value outputVal = outputMap[outputVar];
            // Get output result as dense output
            outputVal.CopyTo(outputData, outputIndex, outputNnzCount);
            var outputShape = outputVar.Shape();

            // Output results
            var numOfElementsInSample = outputVar.Shape().TotalSize();
            ulong seqNo = 0;
            for (int seqIndex = 0; seqIndex < outputData.Count; seqIndex++)
            {
                var dataList = outputData[seqIndex];
                var indexList = outputIndex[seqIndex];
                var nnzCountList = outputIndex[seqIndex];

                long index = 0;
                for (int sampleIndex = 0; sampleIndex < nnzCountList.Count; sampleIndex++)
                {
                    var nnzCount = nnzCountList[sampleIndex];
                    for (long i = 0; i < nnzCount; i++)
                    {
                        indexToRank();

                    }

                }
                ulong elementIndex = 0;
                ulong sampleIndex = 0;
                Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                foreach (var data in seq)
                {
                    // a new sample starts.
                    if (elementIndex++ == 0)
                    {
                        Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                    }
                    Console.Write(" " + data);
                    // reach the end of a sample.
                    if (elementIndex == numOfElementsInSample)
                    {
                        Console.WriteLine(".");
                        elementIndex = 0;
                        sampleIndex++;
                    }
                }
                seqNo++;
            }
        }

        static void Main(string[] args)
        {
            EvaluateUsingCreateValue();
            //Console.WriteLine("======== Evaluate V1 Model ========");
            // EvaluateV1ModelUsingNDView();
            //Console.WriteLine("======== Evaluate V2 Model ========");
            //EvaluateV2ModelUsingNDView();
            //Console.WriteLine("======== Evaluate Model Using System Allocated Memory for Output Value ========");
            //EvaluateUsingSystemAllocatedMemory();
        }

        private List<long> IndexToDimensions(long index, NDShape shape)
        {
            var ranks = shape.Rank();
            long count = index;
            var dims = new List<long>();
            for (uint r=0 ; r < ranks; r++)
            {
                dims[r] = count % shape[r];
                count = count / shape[r];
            }
        }

        private long RankToIndex(NDShape shape, List<long> dimensions)
        {
            long index = 0;
            var ranks = shape.Rank();
            long count;
            for (int r = 0; r < ranks; r++)
            {
                count = r == 0 ? 1 : count = count * shape[r - 1];
                index += count * dimensions[r];
            }
        }

        private static void OutputFunctionInfo(global::Function func)
        {
            var uid = func.Uid();
            System.Console.WriteLine("Function id:" + (string.IsNullOrEmpty(uid) ? "(empty)" : uid));
            var name = func.Name();
            System.Console.WriteLine("Function Name:" + (string.IsNullOrEmpty(name) ? "(empty)" : name));

            // Todo: directly return List() or use a wrapper?
            var argList = func.Arguments().ToList();
            Console.WriteLine("Function arguments:");
            foreach (var arg in argList)
            {
                Console.WriteLine("    name=" + arg.Name() + ", kind=" + arg.Kind() + ", DataType=" + arg.GetDataType() + ", TotalSize=" + arg.Shape().TotalSize());
            }

            var outputList = func.Outputs().ToList();
            Console.WriteLine("Function outputs:");
            foreach (var output in outputList)
            {
                Console.WriteLine("    name=" + output.Name() + ", kind=" + output.Kind() + ", DataType=" + output.GetDataType() + ", TotalSize=" + output.Shape().TotalSize());
            }
        }
    }
}
