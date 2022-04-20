// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPTENSORS_HPP
#define GUARD_NEURALNET_POPTENSORS_HPP

#include <map>
#include <memory>
#include <popart/names.hpp>
#include <popart/popx/preparedtensor.hpp>
#include <popart/popx/viewchangers.hpp>

namespace snap {
class Tensor;
} // namespace snap

namespace popart {
class Ir;

namespace popx {

class PopTensors {
public:
  PopTensors(const Ir &);
  // Insert a new snap::Tensor
  void insert(TensorId, const snap::Tensor &);
  // Reuse snap::Tensor "from" as "to"
  void insertAliased(TensorId to, TensorId from);
  // The same as insert but without any checks against the IR
  void insertUnsafe(TensorId id, const snap::Tensor &pt);
  const snap::Tensor &get(TensorId) const;
  const snap::Tensor &getView(TensorId) const;

  bool hasViewChangers(TensorId) const;
  const ViewChangers &getViewChangers(TensorId);
  void setViewChangers(TensorId, const ViewChangers &viewChangers);

  bool contains(TensorId) const;
  const std::map<TensorId, std::shared_ptr<snap::Tensor>> &getTensors() const;

  bool canAlias(TensorId,
                RequireParallelWritable requireParallelWritable) const;

private:
  void verify(TensorId, const snap::Tensor &);

  std::map<TensorId, std::shared_ptr<snap::Tensor>> tensors_;
  std::map<TensorId, std::shared_ptr<snap::Tensor>> views_;
  std::map<TensorId, std::shared_ptr<ViewChangers>> viewChangers_;
  const Ir &ir;
};

} // namespace popx
} // namespace popart

#endif
