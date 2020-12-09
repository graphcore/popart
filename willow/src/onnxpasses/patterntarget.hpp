// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_PATTERNTARGET_HPP
#define GUARD_NEURALNET_ONNXTOONNX_PATTERNTARGET_HPP

#include <array>
#include <onnx/onnx_pb.h>
#include <onnxpasses/suffixer.hpp>
#include <string>
#include <poprithms/ndarray/shape.hpp>

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

  // TODO(T31464)
  // std::map<std::string, poprithms::ndarray::Shape> shapes;
};
} // namespace onnxpasses
} // namespace popart

#endif
