// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_PATTERNTARGET_HPP
#define GUARD_NEURALNET_ONNXTOONNX_PATTERNTARGET_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxpasses/suffixer.hpp>
#include <string>
#include <poprithms/ndarray/shape.hpp>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {

class PatternTarget {

public:
  PatternTarget(GraphProto &);

  /**
   * The ONNX Graph which Patterns modify. It is the user's responsibility to
   * ensure that the Graph is not deleted before this PatternTarget is.
   * */
  GraphProto &g;

  // The nodes of GraphProto g (above).
  decltype(g.mutable_node()) nodes;

  // An object shared by multiple NodePatterns, used to
  // ensure that unique Tensor names are generated for
  // intermediate Tensors.
  Suffixer suffixer;

  poprithms::ndarray::Shape shape(const std::string &) const;
  std::shared_ptr<Constants> constants() { return foldConstants; }

  uint64_t nShapes() const { return shapes.size(); }

private:
  // Shapes of Tensors in the GraphProto, g.
  std::map<std::string, poprithms::ndarray::Shape> shapes;

  // Constant Tensors obtained by constant folding of GraphProto, g.
  std::shared_ptr<Constants> foldConstants;

  // Methods for taking Tensor Shapes from the GraphProto, and putting them into
  // the shapes map.
  using ValueInfoProtos = decltype(g.value_info());
  void appendShapes(ValueInfoProtos &);
  void extractShapes();
};
} // namespace onnxpasses
} // namespace popart

#endif
