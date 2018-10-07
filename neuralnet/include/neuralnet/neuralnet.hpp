#ifndef GUARD_NEURALNET_NEURALNET_HPP
#define GUARD_NEURALNET_NEURALNET_HPP

namespace neuralnet{

class NeuralNet {
public:
  NeuralNet(std::string logDir);
  // create Graph
  void makeGraph();
  // connect to a backend
  void connect(std::string backend);
  // compile for backend
  void compile();

private:
};

}

#endif
