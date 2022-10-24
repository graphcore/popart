// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MATMULX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MATMULX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <set>
#include <snap/Tensor.hpp>
#include <utility>
#include <vector>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace poplar {
class OptionFlags;
} // namespace poplar
namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class MatMulOp;
class MatMulBaseOp;
class Op;
namespace popx {
class Devicex;
} // namespace popx

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

  // Returns a tensor of type quarter from an unsigned char
  // tensor of FP8 data in `format`. The metadata tensor is
  // populated with `log2scale` as the scale.
  poplar::Tensor
  prepareQuarterInputTensor(DataType format,
                            poplar::Tensor &x,
                            poplar::Tensor &log2Scale,
                            poplar::program::Sequence &prog) const;

  // Create a poplar program that throws an error if log2scale tensor
  // is not in the range [lower, upper], and does nothing otherwise
  poplar::program::Sequence
  createAssertLog2ScaleInRangeProg(poplar::Tensor &log2ScaleTensor,
                                   int lower,
                                   int upper) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MATMULX_HPP_
