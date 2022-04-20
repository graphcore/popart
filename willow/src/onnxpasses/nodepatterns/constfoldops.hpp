// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_CONSTFOLDOPS_HPP
#define GUARD_NEURALNET_ONNXTOONNX_CONSTFOLDOPS_HPP

#include <string>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {

// Base class for ops to be processed by constant folding.
class ConstFoldOp {
public:
  ConstFoldOp(const std::string &type) : constType(type) {}
  virtual ~ConstFoldOp() = default;

  /**
   * Constant fold the Tensors \a input of the NodeProto
   * \a node, returning the values of the outputs of \a node.
   */
  virtual Constants fold(const NodeProto &node, const Constants &inputs) = 0;
  std::string constPatternType() const { return constType; }

private:
  std::string constType;
};

class AbsCFold : public ConstFoldOp {
public:
  AbsCFold() : ConstFoldOp("Abs") {}
  virtual ~AbsCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class BinaryCFold : public ConstFoldOp {
public:
  BinaryCFold() : ConstFoldOp("Binary") {}
  virtual ~BinaryCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class CastCFold : public ConstFoldOp {
public:
  CastCFold() : ConstFoldOp("Cast") {}
  virtual ~CastCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ConcatCFold : public ConstFoldOp {
public:
  ConcatCFold() : ConstFoldOp("Concat") {}
  virtual ~ConcatCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ConstantCFold : public ConstFoldOp {
public:
  ConstantCFold() : ConstFoldOp("Constant") {}
  virtual ~ConstantCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ConstantOfShapeCFold : public ConstFoldOp {
public:
  ConstantOfShapeCFold() : ConstFoldOp("ConstantOfShape") {}
  virtual ~ConstantOfShapeCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ExpCFold : public ConstFoldOp {
public:
  ExpCFold() : ConstFoldOp("Exp") {}
  virtual ~ExpCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ExpandCFold : public ConstFoldOp {
public:
  ExpandCFold() : ConstFoldOp("Expand") {}
  virtual ~ExpandCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class FloorCFold : public ConstFoldOp {
public:
  FloorCFold() : ConstFoldOp("Floor") {}
  virtual ~FloorCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class FmodCFold : public ConstFoldOp {
public:
  FmodCFold() : ConstFoldOp("Fmod") {}
  virtual ~FmodCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class GatherCFold : public ConstFoldOp {
public:
  GatherCFold() : ConstFoldOp("Gather") {}
  virtual ~GatherCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class IdentityCFold : public ConstFoldOp {
public:
  IdentityCFold() : ConstFoldOp("Identity") {}
  virtual ~IdentityCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class IdentityLossCFold : public ConstFoldOp {
public:
  IdentityLossCFold() : ConstFoldOp("IdentityLoss") {}
  virtual ~IdentityLossCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class NegCFold : public ConstFoldOp {
public:
  NegCFold() : ConstFoldOp("Neg") {}
  virtual ~NegCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ReciprocalCFold : public ConstFoldOp {
public:
  ReciprocalCFold() : ConstFoldOp("Reciprocal") {}
  virtual ~ReciprocalCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ReduceProdCFold : public ConstFoldOp {
public:
  ReduceProdCFold() : ConstFoldOp("ReduceProd") {}
  virtual ~ReduceProdCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ReluCFold : public ConstFoldOp {
public:
  ReluCFold() : ConstFoldOp("Relu") {}
  virtual ~ReluCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ReshapeCFold : public ConstFoldOp {
public:
  ReshapeCFold() : ConstFoldOp("Reshape") {}
  virtual ~ReshapeCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class ScaleCFold : public ConstFoldOp {
public:
  ScaleCFold() : ConstFoldOp("Scale") {}
  virtual ~ScaleCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

/// ShapeCFold takes a tensor as input and outputs a 1D int64 tensor containing
/// the shape of the input tensor.
/// Example: For input tensor of shape (2,3,5) output tensor is {2,3,5}.
class ShapeCFold : public ConstFoldOp {
public:
  ShapeCFold() : ConstFoldOp("Shape") {}
  virtual ~ShapeCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class SliceCFold : public ConstFoldOp {
public:
  SliceCFold() : ConstFoldOp("Slice") {}
  virtual ~SliceCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class SqueezeCFold : public ConstFoldOp {
public:
  SqueezeCFold() : ConstFoldOp("Squeeze") {}
  virtual ~SqueezeCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class TransposeCFold : public ConstFoldOp {
public:
  TransposeCFold() : ConstFoldOp("Transpose") {}
  virtual ~TransposeCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

class UnsqueezeCFold : public ConstFoldOp {
public:
  UnsqueezeCFold() : ConstFoldOp("Unsqueeze") {}
  virtual ~UnsqueezeCFold() = default;

  Constants fold(const NodeProto &, const Constants &) final;
};

} // namespace onnxpasses
} // namespace popart

#endif
