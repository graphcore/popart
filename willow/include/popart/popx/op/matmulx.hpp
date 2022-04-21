// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MATMULX_HPP
#define GUARD_NEURALNET_MATMULX_HPP

#include <popart/names.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
class MatMulOp;
class MatMulBaseOp;
enum class MatMulPartialsType;
namespace popx {

class MatMulOpx : public PopOpx {
public:
  MatMulOpx(Op *, Devicex *);
  ~MatMulOpx() override = default;

  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex index0) const final;

  MatMulOp *getMatMulOp() const;
  void grow(snap::program::Sequence &) const final;

  poplar::Type getOutputType(const snap::Tensor &output) const;
  static std::vector<std::size_t> onnxShapeToPoplar(const Shape &shape);
  static void appendPoplarOptionsForOp(const MatMulBaseOp &op,
                                       poplar::OptionFlags &opts);
  static void addPartialsType(const MatMulPartialsType &partialsType,
                              poplar::OptionFlags &opts);

  static std::pair<snap::Tensor, snap::Tensor>
  groupedMatMulInputsFromOpxInputs(MatMulBaseOp &matmul,
                                   snap::Tensor lhs,
                                   snap::Tensor rhs);

  // Check that mamtul pre-planning has worked, and that growing the matmul
  // operation has not added unexpected entries to the planning cache. Note:
  // poplibs matmul creates a 'joint plan' - a plan for the corresponding
  // fwd, bwd and wu matmuls - all at once if the 'fullyConnectedPass' option
  // is 'TRAINING_*'.
  // But if only a subset of these ops exist in our graph, then only a subset
  // of their plans will exist in the cache. In this case we can expect to see
  // up to 2 more plans generated than expected.
  void verifyCacheSizeUnchanged(size_t beforeCacheSize) const;

private:
  // The ONNX tensor shape
  std::vector<std::size_t> getOutputShape() const;
};

} // namespace popx
} // namespace popart

#endif
