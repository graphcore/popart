#include <neuralnet/error.hpp>
#include <neuralnet/filereader.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/nll.hpp>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {

  // The program takes 1 argument, which is the directory to write logs to.
  // This directory should already exist. The Engine log is engout.log
  if (argc != 2) {
    std::stringstream ss;
    ss << "expected exactly 1 argument: "
       << "the directory to read models (.onnx file) "
       << "and input and output files (.pb files) "
       << "and write logs to. Number of args: " << argc - 1;
    throw neuralnet::error(ss.str());
  }

  // now, argv[1] is the path to the directory where logs are to be written
  // here we just expand it to its canonical form (not strictly needed)
  std::string canLogDir = neuralnet::io::getCanonicalDirName(argv[1]);

  // note : must be the same as in pydriver.py
  auto modelPath = neuralnet::io::appendDirFn(canLogDir, "model.onnx");
  auto dummyTensorPath = neuralnet::io::appendDirFn(canLogDir, "input_0.pb");

  std::vector<std::string> constTensors{};
  neuralnet::Recorder recorder{};
  neuralnet::Schedule schedule{};

  std::cout << "modelPath = " << modelPath << std::endl;
  auto model = neuralnet::io::getModel(modelPath);
  std::cout << "model loaded" << std::endl;
  onnx::TensorProto tensor = neuralnet::io::getTensor(dummyTensorPath);
  std::cout << "tensor loaded" << std::endl;

  neuralnet::PreRunKnowledge preRunKnowledge{};
  neuralnet::TensorInfo dataInputInfo(tensor);
  preRunKnowledge.addInfo(model.graph().input(0).name(), dataInputInfo);


  // the scalar label
  int batch_size = dataInputInfo.dim(0);
  neuralnet::TensorId labelTensorId = "label";
  neuralnet::TensorInfo labelInputInfo(neuralnet::TP::INT32, {batch_size});
  preRunKnowledge.addInfo(labelTensorId, labelInputInfo);

  auto loss = std::unique_ptr<neuralnet::Loss>(
      new neuralnet::NegLogLikeLoss(model, labelTensorId));

  std::vector<std::unique_ptr<neuralnet::Regularizer>> regularizers;

  neuralnet::Graph graph(std::move(model),
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
