
//
// This example demonstrates how to create a custom operator for onnx, in this
// case an op that will take a tensor and cube all the elements
//
//

#include "popart/opmanager.hpp"
#include <iostream>

auto main(int argc, char **argv) -> int {

  // TODO : parse input arguments so we can test on different targets cpu vs hw
  (void)argc;
  (void)argv;

  auto ops = popart::OpManager::getSupportedOperations(false);

  std::cout << "Supported ONNX Operators (" << popart::Domain::ai_onnx << ")"
            << std::endl;
  std::cout << "===================================================="
            << std::endl;
  for (auto &op : ops) {
    if (op.domain == popart::Domain::ai_onnx)
      std::cout << op << std::endl;
  }

  std::cout << std::endl;

  std::cout << "Custom ONNX Operators (" << popart::Domain::ai_graphcore << ")"
            << std::endl;
  std::cout << "===================================================="
            << std::endl;
  for (auto &op : ops) {
    if (op.domain == popart::Domain::ai_graphcore)
      std::cout << op << std::endl;
  }
}
