// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <onnxpasses/nodepattern.hpp>
#include <ostream>
#include <popart/error.hpp>

namespace popart {
namespace onnxpasses {

using Node = ONNX_NAMESPACE::NodeProto;

bool NodePattern::run(const Node &node) {
  auto change = go(node);
  nHits_ += change;
  return change;
}

Node &NodePattern::copy(const Node &toCopy) {
  auto n = nodes->Add();
  *n     = toCopy;
  return *n;
}

Node &NodePattern::binary(const Node &toCopy,
                          const std::array<std::string, 2> &ins,
                          const std::string &out,
                          const std::string &type) {
  auto &n = *nodes->Add();
  n       = toCopy;
  setIO(n, {std::get<0>(ins), std::get<1>(ins)}, {out});
  n.set_op_type(type);
  return n;
}

Node &NodePattern::copyUnderscorePrefixedAttributes(const Node &src) {
  auto &n = *nodes->Add();
  for (const auto &srcAtt : src.attribute()) {
    if (srcAtt.name().find("__") == 0) {
      auto pn = n.add_attribute();
      *pn     = srcAtt;
    }
  }
  return n;
}

Node &NodePattern::unary(const Node &toCopy,
                         const std::string &in_,
                         const std::string &out,
                         const std::string &type) {
  auto &n = *nodes->Add();
  n       = toCopy;
  setIO(n, {in_}, {out});
  n.set_op_type(type);
  return n;
}

Node &NodePattern::binaryConstScalar(const Node &toCopy,
                                     const std::string &inName,
                                     const std::string &outName,
                                     const std::string &binaryOpName,
                                     ScalarInIndex inIndex,
                                     float v) {
  auto &n = unary(toCopy, inName, outName, "BinaryConstScalar");
  n.set_domain("ai.graphcore");

  // scalar value
  auto &att0 = *n.add_attribute();
  att0.set_name("value");
  att0.set_f(v);

  // binary operation
  auto &att1 = *n.add_attribute();
  att1.set_name("op");
  att1.set_s(binaryOpName);

  // scalar input index
  auto &att2 = *n.add_attribute();
  att2.set_name("scalar_in_index");
  att2.set_i(inIndex == ScalarInIndex::One ? 1ll : 0ll);

  return n;
}

Node &NodePattern::setIO(Node &n,
                         const std::vector<std::string> &inNames,
                         const std::vector<std::string> &outNames) {

  n.clear_input();
  for (auto inName : inNames) {
    n.add_input(inName);
  }

  n.clear_output();
  for (auto outName : outNames) {
    n.add_output(outName);
  }
  return n;
}

} // namespace onnxpasses
} // namespace popart
