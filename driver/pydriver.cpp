#include <neuralnet/error.hpp>
#include <neuralnet/filereader.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/loss.hpp>
#include <neuralnet/nll.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

int main(int argc, char **argv) {

  using namespace neuralnet;

  // The program takes 1 argument, which is the directory to write logs to.
  // This directory should already exist. The Engine log is engout.log
  if (argc != 2) {
    std::stringstream ss;
    ss << "expected exactly 1 argument: "
       << "the directory to read models (.onnx file) "
       << "and input and output files (.pb files) "
       << "and write logs to. Number of args: " << argc - 1;
    throw error(ss.str());
  }

  // now, argv[1] is the path to the directory where logs are to be written
  // here we just expand it to its canonical form (not strictly needed)
  std::string canLogDir = io::getCanonicalDirName(argv[1]);

  // note : must be the same as in pydriver.py
  auto modelPath = io::appendDirFn(canLogDir, "model.onnx");
  auto inNamesPath = io::appendDirFn(canLogDir, "input_names.txt");
  auto outNamesPath = io::appendDirFn(canLogDir, "output_names.txt");
  auto lossPath = io::appendDirFn(canLogDir, "losses.txt");
  io::confirmRegularFile(inNamesPath);
  io::confirmRegularFile(outNamesPath);


  // load inputs
  PreRunKnowledge preRunKnowledge{};
  std::map<std::string, onnx::TensorProto> inputs;
  std::string inputName;
  std::ifstream input(inNamesPath, std::ios::in);
  while (std::getline(input, inputName)) {
    auto tensorInPath =
        io::appendDirFn(canLogDir, inputName + ".pb");
        inputs[inputName] = io::getTensor(tensorInPath);
        preRunKnowledge.addInfo(inputName, {inputs[inputName]});
        std::cout << "loaded input " << inputName << std::endl;
  }

  // load outputs
  std::map<std::string, onnx::TensorProto> outputs;
  std::string outputName;
  input = std::ifstream(outNamesPath, std::ios::in);
  while (std::getline(input, outputName)) {
    auto tensorOutPath =
        io::appendDirFn(canLogDir, outputName + ".pb");
        outputs[outputName] = io::getTensor(tensorOutPath);
        std::cout << "loaded output " << outputName << std::endl;
  }

  // load losses
  input = std::ifstream(lossPath, std::ios::in);
  std::string lossLine;
  while (std::getline(input, lossLine)){
    auto found = lossLine.find(':');
    if (found == std::string::npos){
      throw error("invalid loss string");
    }
    auto lossName  = lossLine.substr(0, found);
    auto argsLine = lossLine.substr(found+ 1);
    std::cout << "----> " << lossName << "     " << argsLine << std::endl;

    std::cout << "bam ka bam" << std::endl;
    switch (lossMap().at(lossName)) {
      case eLoss::NLL: {
      throw error("an NLL loss");
    }
      case eLoss::L1: {
      throw error("a L1 loss");
    }
    }
  }

  std::vector<std::string> constTensors{};
  Recorder recorder{};
  Schedule schedule{};

  std::cout << "modelPath = " << modelPath << std::endl;
  auto model = io::getModel(modelPath);


//  the scalar label
//  int batch_size = dataInputInfo.dim(0);
//  TensorId labelTensorId = "labels";
//  TensorInfo labelInputInfo(TP::INT32, {batch_size});
//  preRunKnowledge.addInfo(labelTensorId, labelInputInfo);

  auto loss = std::unique_ptr<Loss>(
      //new NegLogLikeLoss(model, labelTensorId));
      new NegLogLikeLoss("loss", "ponkle")); //;"loss", "labels"));

  std::vector<std::unique_ptr<Regularizer>> regularizers;

  Graph graph(std::move(model),
                         std::move(preRunKnowledge),
                         std::move(recorder),
                         std::move(loss),
                         std::move(regularizers),
                         std::move(schedule),
                         std::move(constTensors));

  std::stringstream ss2;
  graph.append(ss2);
  std::cout << ss2.str(); 
  return 0;

}
