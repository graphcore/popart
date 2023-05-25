// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_NORMALIZE_IMAGEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_NORMALIZE_IMAGEX_HPP_

#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <poplar/DebugContext.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class NormalizeImageOpx : public popart::popx::Opx {
public:
  NormalizeImageOpx(popart::Op *op, popart::popx::Devicex *devicex);

  poplar::Tensor createInput(popart::InIndex index,
                             const poplar::DebugNameAndId &dnai) const override;

  popart::popx::InputCreatorType
  getInputCreatorType(popart::InIndex index) const override;

  std::set<popart::TensorId>
      mustExistBeforeCreate(popart::InIndex) const override;

  poplar::Tensor
  createNormalizedImageInput(const poplar::DebugNameAndId &dnai) const;

  void grow(poplar::program::Sequence &prog) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_NORMALIZE_IMAGEX_HPP_
