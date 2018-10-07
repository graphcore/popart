#include <neuralnet/neuralnet.hpp>

namespace neuralnet {

class NeuralNet {
  NeuralNet::NeuralNet(std::string logDir_) {
    std::cout << "hello from the constructor" << std::endl;
    // port from driver
  }

  void NeuralNet::makeGraph() {
    std::cout << "hello from make graph" << std::endl;
  }

  void NeuralNet::connect(std::string backend) {}

  void NeuralNet::compile() {}
}

#endif
