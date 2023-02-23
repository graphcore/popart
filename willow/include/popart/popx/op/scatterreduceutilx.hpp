// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEUTILX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERREDUCEUTILX_HPP_

#include <cstddef>
#include <vector>

#include "popart/vendored/optional.hpp"

namespace poplar {
class Tensor;
class OptionFlags;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popops {
class SlicePlan;
} // namespace popops

namespace popart {

class ScatterReduceOp;
class ScatterReduceGradOp;

namespace popx {
class Opx;
enum class SlicePlanUsedFor;

namespace scatterreduceutilx {

/**
 * Strategy implementation for calculating the forward and backward passes of
 * the scatter reduction.
 *
 * Uses the template pattern to delegate the details of calling the appropriate
 * poplibs API for the requested reduction mode. This interface will handles the
 * preparation of the broadcasted vs vectorised cases for the index input.
 */
class IScatterReductionStrategy {
public:
  virtual poplar::OptionFlags
  createForwardPlanOptions(const ScatterReduceOp &op) const = 0;
  virtual poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const          = 0;
  virtual void initReductionOutput(const Opx &opx,
                                   const poplar::Tensor &out,
                                   poplar::program::Sequence &prog) const = 0;
  virtual void forward(const ScatterReduceOp &op,
                       const Opx &opx,
                       poplar::Tensor output,
                       poplar::Tensor data,
                       poplar::Tensor indices,
                       std::size_t axis,
                       std::size_t group_size,
                       poplar::program::Sequence &prog,
                       const popops::SlicePlan &plan) const               = 0;
  virtual std::vector<poplar::Tensor>
  backward(const ScatterReduceGradOp &op,
           const Opx &opx,
           poplar::Tensor &gradIn,
           poplar::Tensor &indices,
           size_t axis,
           size_t group_size,
           poplar::program::Sequence &prog,
           const popops::SlicePlan &plan) const = 0;

  virtual ~IScatterReductionStrategy() = default;
};

class ScatterReductionStrategy : public IScatterReductionStrategy {
public:
  virtual ~ScatterReductionStrategy() = default;

private:
  void initReductionOutput(const Opx &opx,
                           const poplar::Tensor &out,
                           poplar::program::Sequence &prog) const override;
  void forward(const ScatterReduceOp &op,
               const Opx &opx,
               poplar::Tensor output,
               poplar::Tensor data,
               poplar::Tensor indices,
               size_t axis,
               size_t group_size,
               poplar::program::Sequence &prog,
               const popops::SlicePlan &plan) const override;
  std::vector<poplar::Tensor>
  backward(const ScatterReduceGradOp &op,
           const Opx &opx,
           poplar::Tensor &gradIn,
           poplar::Tensor &indices,
           size_t axis,
           size_t group_size,
           poplar::program::Sequence &prog,
           const popops::SlicePlan &plan) const override;

  poplar::OptionFlags
  createForwardPlanOptions(const ScatterReduceOp &op) const override final;
  poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const override;

  virtual void applyReduction(const Opx &opx,
                              const poplar::Tensor &target,
                              const poplar::Tensor &update,
                              const poplar::Tensor &indices,
                              poplar::program::Sequence &prog,
                              const popops::SlicePlan &plan,
                              bool isGrouped) const = 0;

  virtual std::vector<poplar::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const Opx &opx,
               const poplar::Tensor &gradIn,
               const poplar::Tensor &indices,
               size_t axis,
               size_t group_size,
               poplar::program::Sequence &prog,
               const popops::SlicePlan &plan) const = 0;

  virtual SlicePlanUsedFor forwardSlicePlanUsedFor() const = 0;
  void prepMultiUpdateBroadcastedTensors(const Opx &opx,
                                         poplar::Tensor &output,
                                         poplar::Tensor &data,
                                         poplar::Tensor &indices,
                                         size_t axis,
                                         size_t group_size,
                                         poplar::program::Sequence &prog) const;
  void prepMultiUpdateTensors(poplar::Tensor &output,
                              poplar::Tensor &data,
                              poplar::Tensor &indices,
                              size_t axis,
                              size_t group_size) const;
  void prepMultiSliceBroadcastedTensors(const Opx &opx,
                                        poplar::Tensor &data,
                                        poplar::Tensor &indices,
                                        size_t axis,
                                        size_t group_size,
                                        poplar::program::Sequence &prog) const;
  void prepMultiSliceTensors(poplar::Tensor &data,
                             poplar::Tensor &indices,
                             size_t axis,
                             size_t group_size) const;
  bool
  shouldShrinkTensor(const std::vector<std::size_t> &tensorShape,
                     const std::vector<std::size_t> &shrinkShape,
                     const nonstd::optional<std::size_t> skipAxis = {}) const;
  poplar::Tensor shrinkTensorToFitShape(
      const poplar::Tensor &tensor,
      const std::vector<std::size_t> &shape,
      const nonstd::optional<std::size_t> skipAxis = {}) const;
  int calcNumDataCols(const poplar::Tensor &tensor, const int startDim) const;
};

class SumReductionStrategy : public ScatterReductionStrategy {
  void applyReduction(const Opx &opx,
                      const poplar::Tensor &target,
                      const poplar::Tensor &update,
                      const poplar::Tensor &indices,
                      poplar::program::Sequence &prog,
                      const popops::SlicePlan &plan,
                      const bool isGrouped) const override;
  std::vector<poplar::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const Opx &opx,
               const poplar::Tensor &gradIn,
               const poplar::Tensor &indices,
               size_t axis,
               size_t group_size,
               poplar::program::Sequence &prog,
               const popops::SlicePlan &plan) const override;
  SlicePlanUsedFor forwardSlicePlanUsedFor() const override;
};

class NoneReductionStrategy : public ScatterReductionStrategy {
  poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const override;

  void applyReduction(const Opx &opx,
                      const poplar::Tensor &target,
                      const poplar::Tensor &update,
                      const poplar::Tensor &indices,
                      poplar::program::Sequence &prog,
                      const popops::SlicePlan &plan,
                      const bool isGrouped) const override;
  std::vector<poplar::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const Opx &opx,
               const poplar::Tensor &gradIn,
               const poplar::Tensor &indices,
               size_t axis,
               size_t group_size,
               poplar::program::Sequence &prog,
               const popops::SlicePlan &plan) const override;
  SlicePlanUsedFor forwardSlicePlanUsedFor() const override;
};

class MaxReductionStrategy : public ScatterReductionStrategy {
protected:
  void applyReduction(const Opx &opx,
                      const poplar::Tensor &target,
                      const poplar::Tensor &update,
                      const poplar::Tensor &indices,
                      poplar::program::Sequence &prog,
                      const popops::SlicePlan &plan,
                      const bool isGrouped) const override;

private:
  void initReductionOutput(const Opx &opx,
                           const poplar::Tensor &out,
                           poplar::program::Sequence &prog) const override;

  std::vector<poplar::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const Opx &opx,
               const poplar::Tensor &gradIn,
               const poplar::Tensor &indices,
               size_t axis,
               size_t group_size,
               poplar::program::Sequence &prog,
               const popops::SlicePlan &plan) const override;
  SlicePlanUsedFor forwardSlicePlanUsedFor() const override;
};

/* TODO(T35696): When poplar supports different reduction modes we should
 * change
 * this class to target that directly rather than going through
 * multiUpdateMax */
class MinReductionStrategy : public MaxReductionStrategy {
protected:
  void applyReduction(const Opx &opx,
                      const poplar::Tensor &target,
                      const poplar::Tensor &update,
                      const poplar::Tensor &indices,
                      poplar::program::Sequence &prog,
                      const popops::SlicePlan &plan,
                      const bool isGrouped) const override;
};

class MulReductionStrategy : public ScatterReductionStrategy {
  void initReductionOutput(const Opx &opx,
                           const poplar::Tensor &out,
                           poplar::program::Sequence &prog) const override;

  void applyReduction(const Opx &opx,
                      const poplar::Tensor &target,
                      const poplar::Tensor &update,
                      const poplar::Tensor &indices,
                      poplar::program::Sequence &prog,
                      const popops::SlicePlan &plan,
                      const bool isGrouped) const override;
  std::vector<poplar::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const Opx &opx,
               const poplar::Tensor &gradIn,
               const poplar::Tensor &indices,
               size_t axis,
               size_t group_size,
               poplar::program::Sequence &prog,
               const popops::SlicePlan &plan) const override;
  SlicePlanUsedFor forwardSlicePlanUsedFor() const override;
};

} // namespace scatterreduceutilx
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERUTILX_HPP_
