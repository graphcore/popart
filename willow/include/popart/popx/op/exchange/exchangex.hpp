// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXCHANGEX_HPP
#define GUARD_NEURALNET_EXCHANGEX_HPP

#include <popart/op/exchange/exchange.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class ExchangeDescriptorx {
public:
  ExchangeDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  virtual ~ExchangeDescriptorx() {}

  void setInTensors(std::vector<std::pair<TensorId, snap::Tensor>> inTensors_) {
    inTensors = inTensors_;
  }

  virtual void pre(snap::Graph &graph,
                   poplar::program::Sequence &prog,
                   poplar::DebugContext context)      = 0;
  virtual void exchange(snap::Graph &graph,
                        poplar::program::Sequence &prog,
                        poplar::DebugContext context) = 0;
  virtual void post(snap::Graph &graph,
                    poplar::program::Sequence &prog,
                    poplar::DebugContext context)     = 0;

  virtual std::vector<snap::Tensor> getOutTensors() const { return outTensors; }

protected:
  Devicex *dv_p;

  std::vector<std::pair<TensorId, snap::Tensor>> inTensors;
  std::vector<snap::Tensor> outTensors;

  ExchangeDescriptor descriptor;
};

class HostLoadDescriptorx : public ExchangeDescriptorx {
public:
  HostLoadDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  virtual void pre(snap::Graph &graph,
                   poplar::program::Sequence &prog,
                   poplar::DebugContext context);
  virtual void exchange(snap::Graph &graph,
                        poplar::program::Sequence &prog,
                        poplar::DebugContext context);
  virtual void post(snap::Graph &graph,
                    poplar::program::Sequence &prog,
                    poplar::DebugContext context);
};

class HostStoreDescriptorx : public ExchangeDescriptorx {
public:
  HostStoreDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  void pre(snap::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            poplar::program::Sequence &prog,
            poplar::DebugContext context) override;
};

class RemoteLoadDescriptorx : public ExchangeDescriptorx {
public:
  RemoteLoadDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  void pre(snap::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            poplar::program::Sequence &prog,
            poplar::DebugContext context) override;
};

class RemoteStoreDescriptorx : public ExchangeDescriptorx {
public:
  RemoteStoreDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  void pre(snap::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            poplar::program::Sequence &prog,
            poplar::DebugContext context) override;
};

std::unique_ptr<ExchangeDescriptorx>
getExchangeDescriptorx(Devicex *dv_p, ExchangeDescriptor descriptor);

class ExchangeBaseOpx : public PopOpx {
public:
  ExchangeBaseOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
