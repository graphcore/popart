// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepattern.hpp>

#include "onnxpasses/onnxnames.hpp"
#include "onnxpasses/patterntarget.hpp"
#include "onnxpasses/suffixer.hpp"

namespace popart {
namespace onnxpasses {

poprithms::ndarray::Shape NodePattern::shape(const std::string &s) const {
  return target->shape(s);
}

bool NodePattern::run(const NodeProto &node) {
  auto change = go(node);
  nHits_ += change;
  return change;
}

NodeProto &NodePattern::copy(const NodeProto &toCopy) {
  auto &n   = add();
  auto nAdd = &n;
  *nAdd     = toCopy;
  return n;
}

NodeProto &NodePattern::binary(const NodeProto &toCopy,
                               const std::array<std::string, 2> &ins,
                               const std::string &out,
                               const std::string &type) {
  auto &n = add();
  n       = toCopy;
  setIO(n, {std::get<0>(ins), std::get<1>(ins)}, {out});
  n.set_op_type(type);
  return n;
}

NodeProto &NodePattern::add() {
  auto ptr = target->nodes->Add();
  return *ptr;
}

std::string NodePattern::withUniqueSuffix(const std::string &n) const {
  return n + target->suffixer.getAndIncr();
}

void NodePattern::copyUnderscorePrefixedAttributes(const NodeProto &src,
                                                   NodeProto &dst) {
  for (const auto &srcAtt : src.attribute()) {
    if (srcAtt.name().find("__") == 0) {
      auto pn = dst.add_attribute();
      *pn     = srcAtt;
    }
  }
}

NodeProto &NodePattern::copyUnderscorePrefixedAttributes(const NodeProto &src) {
  auto &n = add();
  copyUnderscorePrefixedAttributes(src, n);
  return n;
}

NodeProto &NodePattern::unary(const NodeProto &toCopy,
                              const std::string &in_,
                              const std::string &out,
                              const std::string &type) {
  auto &n = add();
  n       = toCopy;
  setIO(n, {in_}, {out});
  n.set_op_type(type);
  return n;
}

NodeProto &NodePattern::binaryConstScalar(const NodeProto &toCopy,
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

void NodePattern::addIntsAttribute(NodeProto &node,
                                   const std::string &name,
                                   const std::vector<int64_t> &value) {
  auto attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(ONNX_NAMESPACE::AttributeProto::INTS);
  for (int64_t i : value) {
    attr->add_ints(i);
  }
}

NodeProto &NodePattern::setIO(NodeProto &n,
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
