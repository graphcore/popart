// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_CONSTFOLDOPS_HPP
#define GUARD_NEURALNET_ONNXTOONNX_CONSTFOLDOPS_HPP

#include <onnxpasses/nodepattern.hpp>

namespace popart {
namespace onnxpasses {

// Base class for ops to be processed by constant folding.
class ConstFoldOp {
public:
  ConstFoldOp(const std::string &type) : constType(type) {}

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

  Constants fold(const NodeProto &, const Constants &) final;
};

class BinaryCFold : public ConstFoldOp {
public:
  BinaryCFold() : ConstFoldOp("Binary") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class CastCFold : public ConstFoldOp {
public:
  CastCFold() : ConstFoldOp("Cast") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ConcatCFold : public ConstFoldOp {
public:
  ConcatCFold() : ConstFoldOp("Concat") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ConstantCFold : public ConstFoldOp {
public:
  ConstantCFold() : ConstFoldOp("Constant") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ConstantOfShapeCFold : public ConstFoldOp {
public:
  ConstantOfShapeCFold() : ConstFoldOp("ConstantOfShape") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ExpCFold : public ConstFoldOp {
public:
  ExpCFold() : ConstFoldOp("Exp") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ExpandCFold : public ConstFoldOp {
public:
  ExpandCFold() : ConstFoldOp("Expand") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class FloorCFold : public ConstFoldOp {
public:
  FloorCFold() : ConstFoldOp("Floor") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class FmodCFold : public ConstFoldOp {
public:
  FmodCFold() : ConstFoldOp("Fmod") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class GatherCFold : public ConstFoldOp {
public:
  GatherCFold() : ConstFoldOp("Gather") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class IdentityCFold : public ConstFoldOp {
public:
  IdentityCFold() : ConstFoldOp("Identity") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class IdentityLossCFold : public ConstFoldOp {
public:
  IdentityLossCFold() : ConstFoldOp("IdentityLoss") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class NegCFold : public ConstFoldOp {
public:
  NegCFold() : ConstFoldOp("Neg") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ReciprocalCFold : public ConstFoldOp {
public:
  ReciprocalCFold() : ConstFoldOp("Reciprocal") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ReluCFold : public ConstFoldOp {
public:
  ReluCFold() : ConstFoldOp("Relu") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ReshapeCFold : public ConstFoldOp {
public:
  ReshapeCFold() : ConstFoldOp("Reshape") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class ScaleCFold : public ConstFoldOp {
public:
  ScaleCFold() : ConstFoldOp("Scale") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class SliceCFold : public ConstFoldOp {
public:
  SliceCFold() : ConstFoldOp("Slice") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class SqueezeCFold : public ConstFoldOp {
public:
  SqueezeCFold() : ConstFoldOp("Squeeze") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class TransposeCFold : public ConstFoldOp {
public:
  TransposeCFold() : ConstFoldOp("Transpose") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

class UnsqueezeCFold : public ConstFoldOp {
public:
  UnsqueezeCFold() : ConstFoldOp("Unsqueeze") {}

  Constants fold(const NodeProto &, const Constants &) final;
};

} // namespace onnxpasses
} // namespace popart

#endif
