// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_ONNXTOONNX_HPP
#define GUARD_NEURALNET_ONNXTOONNX_ONNXTOONNX_HPP
#include <onnx/onnx_pb.h>
#include <onnxpasses/onnxnames.hpp>

namespace popart {
namespace onnxpasses {

/**
 * Abstract base class for onnx -> onnx transformations
 * */
class IOnnxToOnnx {

public:
  IOnnxToOnnx();
  virtual ~IOnnxToOnnx();

  /** Inplace modification of a ONNX GraphProto. */
  virtual void canonnxalize(GraphProto &) const = 0;

  /** Create a copy of \a gIn, modify it with canonnxalize, and return the
   * modified GraphProto */
  GraphProto getCanonnxalized(const GraphProto &gIn) const;
};

class Canonnxalizer : public IOnnxToOnnx {

public:
  Canonnxalizer();
  virtual ~Canonnxalizer() override;

  /**
   * An ONNX to ONNX GraphProto transformation which removes or modifies
   * unsupported Nodes and Attributes.
   *
   * Note that certain transformations/patterns cannot be performed here, in the
   * ONNX representation, and these are performed directly on the PopART Ir, at
   * a later stage.
   * */
  virtual void canonnxalize(GraphProto &) const final;
};

} // namespace onnxpasses
} // namespace popart

#endif
