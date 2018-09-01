#include <neuralnet/error.hpp>
#include <neuralnet/filereader.hpp>
#include <neuralnet/graph.hpp>
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

  auto modelPath = neuralnet::io::appendDirFn(canLogDir, "model.onnx");
  std::vector<std::string> constTensors{};
  neuralnet::Recorder recorder{};
  neuralnet::Schedule schedule{};

  std::cout << "modelPath = " << modelPath << std::endl;
  auto model = neuralnet::io::getModel(modelPath);
  std::cout << "model loaded" << std::endl;
  neuralnet::Graph graph(std::move(model),
                         std::move(recorder),
                         std::move(schedule),
                         std::move(constTensors));

  std::stringstream ss;
  graph.append(ss);
  std::cout << ss.str(); 
  return 0;

}
