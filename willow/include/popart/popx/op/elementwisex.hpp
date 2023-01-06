// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ELEMENTWISEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ELEMENTWISEX_HPP_

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <poplar/OptionFlags.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/opx.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "poplar/Tensor.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

// A base class with functions for computing in-place and
// out-of place element-wise unary operations
class EwuComputex {
public:
  EwuComputex()          = default;
  virtual ~EwuComputex() = default;

  virtual poplar::Tensor outplace(poplar::program::Sequence &,
                                  poplar::Graph &,
                                  const poplar::Tensor &,
                                  const poplar::DebugNameAndId &,
                                  const std::string &) const;

  virtual void inplace(poplar::program::Sequence &,
                       poplar::Graph &,
                       const poplar::Tensor &t,
                       const poplar::DebugNameAndId &dnai,
                       const std::string &) const = 0;

  poplar::Tensor cloneNcopy(poplar::program::Sequence &,
                            poplar::Graph &,
                            const poplar::Tensor &,
                            const poplar::DebugNameAndId &) const;

  // certain ops reshape the input tensor (eg Softmax and LogSoftmax)
  virtual poplar::Tensor reshape(const poplar::Tensor &t) const { return t; }

  static poplar::Tensor coerceTo2D(const poplar::Tensor &t, int64_t axis);
};

// Base class for elementwise unary operations
class ElementWiseUnaryOpx : public Opx {
public:
  ElementWiseUnaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

// non-inplace elementwise unary operations
class ElementWiseUnaryOutplaceOpx : public ElementWiseUnaryOpx {
public:
  ElementWiseUnaryOutplaceOpx(Op *,
                              Devicex *,
                              std::unique_ptr<EwuComputex> cx_);
  void grow(poplar::program::Sequence &) const final;

private:
  std::unique_ptr<EwuComputex> cx;
};

// inplace elementwise unary operations
class ElementWiseUnaryInplaceOpx : public ElementWiseUnaryOpx {
public:
  ElementWiseUnaryInplaceOpx(Op *op,
                             Devicex *devx,
                             std::unique_ptr<EwuComputex> cx_)
      : ElementWiseUnaryOpx(op, devx), cx(std::move(cx_)) {}
  void grow(poplar::program::Sequence &prog) const final;

private:
  std::unique_ptr<EwuComputex> cx;
};

// Base class for computing either in-place or out-of-place elementwise binary
// operations. There are various factors that determine whether the operation
// will be evaluated in-place:
//   * An Opx implementation can "opt out" of in-placing so that it is always
//   evaluated out-of-place. This is expected to be known at compile-time.
//   * At runtime, supported operations may be evaluated in-place when the
//   input-output tensor isParallelWriteable.
//   * The current tile imbalance at runtime is also compared to a fixed
//   threshold to determine whether to compute supported operations in-place.
class EwbComputex {
public:
  // The inplacing policy that this class will use.
  enum class InplacePolicy {
    // Never evaluate the operation in-place
    NEVER,
    // Possibly evaluate the operation with the LHS (input arg 0) in-place
    LHS,
    // Possibly evaluate the operation with the RHS (input arg 1) in-place
    RHS
  };

  explicit EwbComputex(InplacePolicy ip = InplacePolicy::NEVER)
      : inplacePolicy(ip) {}

  virtual ~EwbComputex() = default;

  // Check whether this class might support in-place evaluation
  bool inplaceSupported() const;

  // Get the InIndex for the argument that will be in-placed
  // e.g. Using InplacePolicy::LHS -> 0
  //            InplacePolicy::RHS -> 1
  // An internal_error is raised when this class is configured with
  // InplacePolicy::NEVER
  InIndex getInplaceArgInIndex() const;

  // Get the InIndex for the argument that will be out-of-place
  // e.g. Using InplacePolicy::LHS -> 1
  //            InplacePolicy::RHS -> 0
  // An internal_error is raised when this class is configured with
  // InplacePolicy::NEVER
  InIndex getOutplaceArgInIndex() const;

  // Evaluate the operation out-of-place
  virtual poplar::Tensor outplace(poplar::program::Sequence &,
                                  poplar::Graph &,
                                  const poplar::Tensor &,
                                  const poplar::Tensor &,
                                  const poplar::DebugNameAndId &,
                                  const std::string &) const = 0;

  // Evaluate the operation in-place if possible
  virtual poplar::Tensor maybeInplace(poplar::program::Sequence &,
                                      poplar::Graph &,
                                      poplar::Tensor &,
                                      poplar::Tensor &,
                                      const poplar::DebugNameAndId &,
                                      const std::string &) const = 0;

  poplar::Tensor mapMaybeInPlace(poplar::Graph &graph,
                                 popops::expr::BinaryOpType op,
                                 poplar::Tensor &tInOut,
                                 poplar::Tensor &tIn,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugContext &debugContext,
                                 const poplar::OptionFlags &options = {},
                                 const std::string &name            = "") const;

private:
  InplacePolicy inplacePolicy;
};

// Base class for elementwise binary operations
class ElementWiseBinaryOpx : public Opx {
public:
  ElementWiseBinaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const override;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const override;
  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

// non-inplace elementwise binary operations
class ElementWiseBinaryOutplaceOpx : public ElementWiseBinaryOpx {
public:
  ElementWiseBinaryOutplaceOpx(Op *op,
                               Devicex *devx,
                               std::unique_ptr<EwbComputex> cx_)
      : ElementWiseBinaryOpx(op, devx), cx(std::move(cx_)) {}

  void grow(poplar::program::Sequence &) const final;

private:
  std::unique_ptr<EwbComputex> cx;
};

// inplace elementwise binary operations
class ElementWiseBinaryInplaceOpx : public ElementWiseBinaryOpx {
public:
  ElementWiseBinaryInplaceOpx(Op *op,
                              Devicex *devx,
                              std::unique_ptr<EwbComputex> cx_)
      : ElementWiseBinaryOpx(op, devx), cx(std::move(cx_)) {}

  void grow(poplar::program::Sequence &) const final;

private:
  std::unique_ptr<EwbComputex> cx;
};

// Base class for binary comparison operations
class BinaryComparisonOpx : public Opx {
public:
  BinaryComparisonOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ELEMENTWISEX_HPP_
