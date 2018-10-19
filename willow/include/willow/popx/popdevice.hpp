#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplar/DeviceManager.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

#include <willow/device.hpp>

namespace willow {

class PopOp;

class PopDevice : public Device {

public:
  PopDevice(const Ir *);
  virtual void prepare() override final;
  PopOp * getPopOp(OpId);

private:
  std::unique_ptr<poplar::Graph> pGraph {nullptr};
  std::unique_ptr<poplar::Engine> pEngine {nullptr};
  std::unique_ptr<poplar::Target> pTarget {nullptr};
  poplar::Device popDevice;

  poplar::program::Sequence weightsToHost;
  poplar::program::Sequence optimizerToHost;
  poplar::program::Sequence weightsFromHost;
  poplar::program::Sequence step;

  poplar::Tensor createPopTensor(Tensor * tensor);
  std::unique_ptr<PopOp> createPopOp(const Op *);

  // 1-to-1 mapping between Ops and PopOps
  std::map<OpId, std::unique_ptr<PopOp>> pop_ops;
  std::map<TensorId, poplar::Tensor> pop_tensors;

};

} // namespace willow

#endif
