// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP

#include <popart/popx/debugcontextx.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

// A base class with functions for computing in-place and
// out-of place element-wise unary operations
class EwuComputex {
public:
  EwuComputex()          = default;
  virtual ~EwuComputex() = default;

  virtual snap::Tensor outplace(snap::program::Sequence &,
                                snap::Graph &,
                                const snap::Tensor &,
                                const poplar::DebugNameAndId &,
                                const std::string &) const;

  virtual void inplace(snap::program::Sequence &,
                       snap::Graph &,
                       const snap::Tensor &t,
                       const poplar::DebugNameAndId &,
                       const std::string &) const = 0;

  snap::Tensor cloneNcopy(snap::program::Sequence &,
                          snap::Graph &,
                          const snap::Tensor &,
                          const poplar::DebugNameAndId &) const;

  // certain ops reshape the input tensor (eg Softmax and LogSoftmax)
  virtual snap::Tensor reshape(const snap::Tensor &t) const { return t; }

  static snap::Tensor coerceTo2D(const snap::Tensor &t, int64_t axis);
};

// Base class for elementwise unary operations
class ElementWiseUnaryOpx : public PopOpx {
public:
  ElementWiseUnaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  snap::Tensor
      unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

// non-inplace elementwise unary operations
class ElementWiseUnaryOutplaceOpx : public ElementWiseUnaryOpx {
public:
  ElementWiseUnaryOutplaceOpx(Op *,
                              Devicex *,
                              std::unique_ptr<EwuComputex> cx_);
  void grow(snap::program::Sequence &) const final;

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
  void grow(snap::program::Sequence &prog) const final;

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
  virtual snap::Tensor outplace(snap::program::Sequence &,
                                snap::Graph &,
                                const snap::Tensor &,
                                const snap::Tensor &,
                                const poplar::DebugNameAndId &,
                                const std::string &) const = 0;

  // Evaluate the operation in-place if possible
  virtual snap::Tensor maybeInplace(snap::program::Sequence &,
                                    snap::Graph &,
                                    const snap::Tensor &,
                                    const snap::Tensor &,
                                    const poplar::DebugNameAndId &,
                                    const std::string &) const = 0;

private:
  InplacePolicy inplacePolicy;
};

// Base class for elementwise binary operations
class ElementWiseBinaryOpx : public PopOpx {
public:
  ElementWiseBinaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const override;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const override;
  snap::Tensor
  unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

// non-inplace elementwise binary operations
class ElementWiseBinaryOutplaceOpx : public ElementWiseBinaryOpx {
public:
  ElementWiseBinaryOutplaceOpx(Op *op,
                               Devicex *devx,
                               std::unique_ptr<EwbComputex> cx_)
      : ElementWiseBinaryOpx(op, devx), cx(std::move(cx_)) {}

  void grow(snap::program::Sequence &) const final;

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

  void grow(snap::program::Sequence &) const final;

private:
  std::unique_ptr<EwbComputex> cx;
};

// Base class for binary comparison operations
class BinaryComparisonOpx : public PopOpx {
public:
  BinaryComparisonOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
