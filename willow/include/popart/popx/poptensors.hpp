#ifndef GUARD_NEURALNET_POPTENSORS_HPP
#define GUARD_NEURALNET_POPTENSORS_HPP

#include <set>

#include <popart/names.hpp>

#include <poplar/Tensor.hpp>

namespace popart {
namespace popx {

class PopTensors {
public:
  PopTensors(const Ir &);
  void insert(TensorId, const poplar::Tensor &);
  const poplar::Tensor &get(TensorId) const;
  bool contains(TensorId) const;
  const std::map<TensorId, poplar::Tensor> &getTensors() const;
  void addAlias(TensorId, TensorId);

private:
  std::map<TensorId, poplar::Tensor> tensors_;
  std::map<TensorId, std::set<TensorId>> tensorAliases_;
  const Ir &ir;
};

} // namespace popx
} // namespace popart

#endif
