// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICBASE_HPP
#define GUARD_NEURALNET_DYNAMICBASE_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

// Class hierarchy
// DynamicBaseOp
//   DynamicSliceBaseOp
//     DynamicSliceOp
//     DynamicUpdateUpdateGradOp
//   DynamicSlicePadGradOp
//   DynamicBinaryBaseOp
//     DynamicTernaryBaseOp
//       DynamicAddOp
//       DynamicUpdateOp
//       DynamicTernaryBaseInplaceOp
//         DynamicAddInplaceOp
//         DynamicUpdateInplaceOp
//     DynamicBinaryBaseInplaceOp
//       DynamicZeroInplaceOp
//     DynamicZeroOp
//     DynamicUpdateToUpdateGradOp
//     DynamicZeroGradOp

namespace popart {
class AliasModel;
class OpSerialiserBase;
struct OperatorIdentifier;

/**
 * Dynamic Base Op
 *
 * Base class for operators acting on a run-time selectable slice of a tensor.
 *
 * The word "dynamic" refers to the fact that the \a index can be specified
 * during runtime, where \a index is the second tensor argument of this operator
 * as specified in \see graphcoreoperators.hpp. The \a axes specifies along
 * which axes the tensor should be sliced. The \a size specifies the size of the
 * slices.
 *
 * A slice along an axis can be defined as by the tuple
 * ( \a start, \a stop, \a step )
 * \a start - will be equal the \a index for the respective axis
 * \a stop - will be equal \a index + \a size for the respective axis
 * \a step - will equal 1
 *
 * Limitations:
 * Assuming we would like to slice A with dimension (4, 3)
 * - Step other than 1 is not supported (i.e. A[::2,:] is not supported)
 * - Negative slicing is not supported (i.e. A[:-1,:] is not supported)
 * - \a stop greater than the size of the axis is not supported
 *  (i.e. A[:5,:] is not supported)
 *
 * Example:
 *     Given a Tensor A with shape (3, 2, 4, 5)
 *     If we specify axes = {1, 3} (i.e. we will slice the first and third axis
 *     [counting from 0]) the operator will operate on
 *     A[:, index[0]:(index[0]+size[0]), :, index[1]:(index[1]+size[1])]
 *     If we instead
 *     specify axes = {0, 1, 3} the operator will operate on
 *     A[index[0]:(index[0]+size[0]),
 *       index[1]:(index[1]+size[1]),
 *       :,
 *       index[2]:(index[2]+size[2])]
 **/
class DynamicBaseOp : public Op {
public:
  DynamicBaseOp(const OperatorIdentifier &_opid,
                std::vector<int64_t> axes_,
                std::vector<int64_t> sizes_,
                bool noOverlap_,
                const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  void setup() override;

  static InIndex getIndexInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  const std::vector<int64_t> &getAxes() const { return axes; }
  void setAxes(const std::vector<int64_t> &x) { axes = x; }

  const std::vector<int64_t> &getSizes() const { return sizes; }
  void setSizes(const std::vector<int64_t> &x) { sizes = x; }

  bool isNotOverlapping() const { return noOverlap; }

  TensorInfo createOutInfo() const;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

protected:
  /// Axes along which the operator slices the input
  std::vector<int64_t> axes;
  /// Number of elements the slice consists of along the corresponding axes
  std::vector<int64_t> sizes;

  // If set to true, then correct gradient backpropagation is only guaranteed if
  // each region in the output tensor has only exactly one populator
  // (operation that writes data to this region).
  // There are no run-time or compile-time checks possible to ensure this.
  bool noOverlap;
};

class DynamicSliceBaseOp : public DynamicBaseOp {
public:
  DynamicSliceBaseOp(const OperatorIdentifier &_opid,
                     std::vector<int64_t> axes_,
                     std::vector<int64_t> sizes_,
                     bool noOverlap_,
                     const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  void setup() final;

  TensorInfo createOutInfo() const;
  static InIndex getInIndex() { return 0; }
};

/**
 * Dynamic Binary Base Op
 *
 * Base class for operators acting on a run-time selectable slice of a tensor.
 * The word "binary" refers to the fact that the operator takes two tensors as
 * input.
 *
 * \see DynamicBaseOp for details
 **/
class DynamicBinaryBaseOp : public DynamicBaseOp {
public:
  DynamicBinaryBaseOp(const OperatorIdentifier &_opid,
                      std::vector<int64_t> axes_,
                      std::vector<int64_t> sizes_,
                      bool noOverlap_,
                      const Op::Settings &settings_,
                      TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const override;
  void setup() final;

  const TensorInfo &getUpdateTensorInfo() const { return updateInInfo; }

  static InIndex getUpdateInIndex() { return 0; }
  static InIndex getIndexInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  virtual void growAliasModel(AliasModel &m) const final;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

protected:
  /// The TensorInfo (data_type, shape and meta_shape) for the update tensor
  TensorInfo updateInInfo;
};

class DynamicBinaryBaseInplaceOp : public DynamicBinaryBaseOp {
public:
  DynamicBinaryBaseInplaceOp(const OperatorIdentifier &_opid,
                             std::vector<int64_t> axes_,
                             std::vector<int64_t> sizes_,
                             bool noOverlap_,
                             const Op::Settings &settings_,
                             TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // This Op aliases and modifies the input
  view::Regions aliases(InIndex, OutIndex) const final;
  view::Regions modifies(InIndex) const final;
};

/**
 * Dynamic Ternary Base Op
 *
 * Base class for operators acting on a run-time selectable slice of a tensor.
 * The word "ternary" refers to the fact that the operator takes three tensors
 * as input.
 *
 * \see DynamicBaseOp for details
 **/
class DynamicTernaryBaseOp : public DynamicBinaryBaseOp {
public:
  DynamicTernaryBaseOp(const OperatorIdentifier &_opid,
                       std::vector<int64_t> axes_,
                       std::vector<int64_t> sizes_,
                       bool noOverlap_,
                       const Op::Settings &settings_,
                       TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const override;
  static InIndex getUpdateInIndex() { return 0; }
  static InIndex getInIndex() { return 2; }
};

class DynamicTernaryBaseInplaceOp : public DynamicTernaryBaseOp {
public:
  DynamicTernaryBaseInplaceOp(const OperatorIdentifier &_opid,
                              std::vector<int64_t> axes_,
                              std::vector<int64_t> sizes_,
                              bool noOverlap_,
                              const Op::Settings &settings_,
                              TensorInfo updateInInfo_ = TensorInfo());
  std::unique_ptr<Op> clone() const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // This Op aliases and modifies the input
  view::Regions aliases(InIndex, OutIndex) const final;
  view::Regions modifies(InIndex) const final;
};

} // namespace popart

#endif
