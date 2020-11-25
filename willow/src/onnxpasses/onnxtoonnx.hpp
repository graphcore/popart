// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_ONNXTOONNX_HPP
#define GUARD_NEURALNET_ONNXTOONNX_ONNXTOONNX_HPP
#include <onnx/onnx_pb.h>

namespace popart {
namespace onnxpasses {

/**
 * Abstract base class for onnx -> onnx transformations
 * */
class IOnnxToOnnx {

public:
  IOnnxToOnnx();
  virtual ~IOnnxToOnnx();

  virtual void canonnxalize(ONNX_NAMESPACE::GraphProto &) const = 0;

  /** Create a copy of \a gIn, modify it with onnxtoonnx, and return the
   * modified GraphProto */
  ONNX_NAMESPACE::GraphProto
  getCanonnxalized(const ONNX_NAMESPACE::GraphProto &gIn) const;
};

class Canonnxalizer : public IOnnxToOnnx {

public:
  Canonnxalizer();
  virtual ~Canonnxalizer() override; // = default;

  /**
   * An ONNX to ONNX Graph transformation which removes or modifies unsupported
   * Nodes and Attributes.
   *
   * Note that certain transformations/patterns cannot be performed in the ONNX
   * representation, and these are performed directly on the PopART Ir, at a
   * later stage.
   * */
  virtual void canonnxalize(ONNX_NAMESPACE::GraphProto &) const final;
};

} // namespace onnxpasses
} // namespace popart

#endif
