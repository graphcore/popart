#ifndef GUARD_NEURALNET_NEURALNET_HPP
#define GUARD_NEURALNET_NEURALNET_HPP

#include <string>

namespace neuralnet {

class Graph;

class NeuralNet {
public:
  NeuralNet(std::string logDir);
  ~NeuralNet();
  // create Graph
  void makeGraph();
  // connect to a backend
  void connect(std::string backend);
  // compile for backend
  void compile();

private:
  std::unique_ptr<Graph> graph;
};

} // namespace neuralnet

#endif
