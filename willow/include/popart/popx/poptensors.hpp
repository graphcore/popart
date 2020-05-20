// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPTENSORS_HPP
#define GUARD_NEURALNET_POPTENSORS_HPP

#include <set>

#include <popart/names.hpp>
#include <popart/popx/viewchangers.hpp>

#include <poplar/Tensor.hpp>

namespace popart {
namespace popx {

class PopTensors {
public:
  PopTensors(const Ir &);
  // Insert a new poplar::Tensor
  void insert(TensorId, const poplar::Tensor &);
  // Reuse poplar::Tensor "from" as "to"
  void insertAliased(TensorId to, TensorId from);
  // The same as insert but without any checks against the IR
  void insertUnsafe(TensorId id, const poplar::Tensor &pt);
  const poplar::Tensor &get(TensorId) const;
  const poplar::Tensor &getView(TensorId) const;

  bool hasViewChangers(TensorId) const;
  const ViewChangers &getViewChangers(TensorId);
  void setViewChangers(TensorId, const ViewChangers &viewChangers);

  bool contains(TensorId) const;
  const std::map<TensorId, std::shared_ptr<poplar::Tensor>> &getTensors() const;

  bool canAlias(TensorId) const;

private:
  void verify(TensorId, const poplar::Tensor &);

  std::map<TensorId, std::shared_ptr<poplar::Tensor>> tensors_;
  std::map<TensorId, std::shared_ptr<poplar::Tensor>> views_;
  std::map<TensorId, std::shared_ptr<ViewChangers>> viewChangers_;
  const Ir &ir;
};

} // namespace popx
} // namespace popart

#endif
