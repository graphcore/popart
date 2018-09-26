#include <fstream>
#include <iostream>
#include <neuralnet/error.hpp>
#include <neuralnet/filereader.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/loss.hpp>
#include <neuralnet/nll.hpp>
#include <neuralnet/l1.hpp>
#include <sstream>

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
  auto modelPath      = io::appendDirFn(canLogDir, "model.onnx");
  auto inNamesPath    = io::appendDirFn(canLogDir, "input_names.txt");
  auto outNamesPath   = io::appendDirFn(canLogDir, "output_names.txt");
  auto lossPath       = io::appendDirFn(canLogDir, "losses.txt");
  auto lossStreamPath = io::appendDirFn(canLogDir, "loss_stream.txt");
  for (auto &x : std::vector<std::string>{
           modelPath, inNamesPath, outNamesPath, lossPath, lossStreamPath}) {
    io::confirmRegularFile(x);
  }

  // load inputs
  PreRunKnowledge preRunKnowledge{};
  std::map<std::string, onnx::TensorProto> inputs;
  std::string inputName;
  std::ifstream input(inNamesPath, std::ios::in);
  while (std::getline(input, inputName)) {
    auto tensorInPath = io::appendDirFn(canLogDir, inputName + ".pb");
    inputs[inputName] = io::getTensor(tensorInPath);
    preRunKnowledge.addInfo(inputName, {inputs[inputName]});
  }

  // load outputs
  std::map<std::string, onnx::TensorProto> outputs;
  std::string outputName;
  input = std::ifstream(outNamesPath, std::ios::in);
  while (std::getline(input, outputName)) {
    auto tensorOutPath  = io::appendDirFn(canLogDir, outputName + ".pb");
    outputs[outputName] = io::getTensor(tensorOutPath);
  }

  // load losses
  std::vector<std::unique_ptr<Loss>> losses;
  input = std::ifstream(lossPath, std::ios::in);
  std::string lossLine;
  while (std::getline(input, lossLine)) {

    auto found = lossLine.find(':');
    if (found == std::string::npos) {
      throw error("invalid loss string in driver");
    }
    auto lossName = lossLine.substr(0, found);
    lossLine = lossLine.substr(found + 1);
    switch (lossMap().at(lossName)) {
    case eLoss::NLL: {
      losses.push_back(std::unique_ptr<Loss>(new NllLoss(lossLine)));
      break;
    };
    case eLoss::L1: {
      losses.push_back(std::unique_ptr<Loss>(new L1Loss(lossLine)));
      break;
    }
    }
  }

  // append pre run knowledge
  std::string infoLine;
  input = std::ifstream(lossStreamPath, std::ios::in);
  while (std::getline(input, infoLine)){

    std::cout << infoLine << std::endl;
    std::istringstream iss(infoLine);
    std::vector<std::string> frags;
    std::string frag;
    while (iss >> frag){
      frags.push_back(frag);
    }
    if (frags.size() != 3){
      throw error("expected [name type shape]");
    }
    std::string tensorName = frags[0];
    TensorInfo info(frags[1], frags[2]);
    preRunKnowledge.addInfo(tensorName, info);

    std::stringstream ss;
    info.append(ss);
    std::cout << ss.str() << std::endl;
  }

  std::vector<std::string> constTensors{};
  Recorder recorder{};
  Schedule schedule{};

  auto model = io::getModel(modelPath);
  std::vector<std::unique_ptr<Regularizer>> regularizers;

  std::cout << "constructing graph" << std::endl;
  Graph graph(std::move(model),
              std::move(preRunKnowledge),
              std::move(recorder),
              std::move(losses),
              std::move(regularizers),
              std::move(schedule),
              std::move(constTensors));

  std::stringstream ss2;
  graph.append(ss2);
  std::cout << ss2.str();
  return 0;
}
