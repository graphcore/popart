
//
// This program will display the list of supported operations, the supported
// input & output tensor types and the support attributes / attribute values. If
// an attribute is '*' all values are supported
//
// If an attribute or input/output is not shown it is not supported
//
//

#include "popart/opmanager.hpp"
#include <iostream>

void showOp(
    std::ostream &os,
    std::pair<const popart::OperatorIdentifier, popart::OpDefinition> op) {
  os << op.first << std::endl;
  if (op.second.inputs.size() > 0) {
    os << " Inputs:" << std::endl;
    for (auto &i : op.second.inputs) {
      os << "  " << i.name << " : Supported types [" << i.supportedTensors
         << "]";
      if (i.constant) {
        os << ", Constant : True";
      }
      os << std::endl;
    }
  }

  if (op.second.outputs.size() > 0) {
    os << " Outputs:" << std::endl;
    for (auto &i : op.second.outputs) {
      os << "  " << i.name << " : Supported types [" << i.supportedTensors
         << "]" << std::endl;
    }
  }

  if (op.second.attributes.size() > 0) {
    os << " Attributes:" << std::endl;
    for (auto &i : op.second.attributes) {
      os << "  " << i.first << " : " << i.second.supportedValuesRegex
         << std::endl;
    }
  }

  os << std::endl;
}

auto main(int argc, char **argv) -> int {

  (void)argc;
  (void)argv;

  auto ops = popart::OpManager::getSupportedOperationsDefinition(false);

  std::cout << "Supported ONNX Operators (" << popart::Domain::ai_onnx << ")"
            << std::endl;
  std::cout << "===================================================="
            << std::endl;

  for (auto &op : ops) {
    if (op.first.domain == popart::Domain::ai_onnx) {
      showOp(std::cout, op);
    }
  }

  std::cout << std::endl;

  std::cout << "Custom ONNX Operators (" << popart::Domain::ai_graphcore << ")"
            << std::endl;
  std::cout << "===================================================="
            << std::endl;

  for (auto &op : ops) {
    if (op.first.domain == popart::Domain::ai_graphcore)
      showOp(std::cout, op);
  }
}
