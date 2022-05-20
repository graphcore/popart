// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXCHANGEX_HPP
#define GUARD_NEURALNET_EXCHANGEX_HPP

#include <memory>
#include <set>
#include <snap/Tensor.hpp>
#include <utility>
#include <vector>
#include <popart/op/exchange/exchange.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace poplar {
enum class FunctionBufferMappingType;
}

namespace popart {
class Op;
class TensorInfo;

namespace popx {
class Devicex;

class ExchangeDescriptorx {
public:
  ExchangeDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  virtual ~ExchangeDescriptorx() {}

  void setInTensors(std::vector<std::pair<TensorId, snap::Tensor>> inTensors_) {
    inTensors = inTensors_;
  }

  virtual void pre(snap::Graph &graph,
                   snap::program::Sequence &prog,
                   poplar::DebugContext context)      = 0;
  virtual void exchange(snap::Graph &graph,
                        snap::program::Sequence &prog,
                        poplar::DebugContext context) = 0;
  virtual void post(snap::Graph &graph,
                    snap::program::Sequence &prog,
                    poplar::DebugContext context)     = 0;

  virtual snap::Tensor unwind(snap::Graph &, snap::Tensor) const;

  /**
   * Create an input tensor that is compatible with host and remote exchange
   * operations without causing rearrangements
   * \param graph Graph on which to create the tensor
   * \param info  Tensor info (data type and shape) of the tensor to create
   * \return      Tensor laid out optimally for exchanges with
   *              createHostTransferableTensor
   */
  virtual snap::Tensor create(snap::Graph &graph, const TensorInfo &info) const;

  virtual std::vector<snap::Tensor> getOutTensors() const { return outTensors; }

  virtual bool rearrangeOnHost() const { return true; }

protected:
  Devicex *dv_p;

  std::vector<std::pair<TensorId, snap::Tensor>> inTensors;
  std::vector<snap::Tensor> outTensors;

  ExchangeDescriptor descriptor;
};

class HostLoadDescriptorx : public ExchangeDescriptorx {
public:
  HostLoadDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  void pre(snap::Graph &graph,
           snap::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                snap::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            snap::program::Sequence &prog,
            poplar::DebugContext context) override;
  snap::Tensor unwind(snap::Graph &, snap::Tensor) const override;
  bool rearrangeOnHost() const override;
};

class HostStoreDescriptorx : public ExchangeDescriptorx {
public:
  HostStoreDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  void pre(snap::Graph &graph,
           snap::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                snap::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            snap::program::Sequence &prog,
            poplar::DebugContext context) override;
  bool rearrangeOnHost() const override;
};

class RemoteLoadDescriptorx : public ExchangeDescriptorx {
public:
  RemoteLoadDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  void pre(snap::Graph &graph,
           snap::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                snap::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            snap::program::Sequence &prog,
            poplar::DebugContext context) override;
  snap::Tensor unwind(snap::Graph &, snap::Tensor) const override;
};

class RemoteStoreDescriptorx : public ExchangeDescriptorx {
public:
  RemoteStoreDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  void pre(snap::Graph &graph,
           snap::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                snap::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            snap::program::Sequence &prog,
            poplar::DebugContext context) override;
};

class ExternalCodeLoadDescriptorx : public ExchangeDescriptorx {
public:
  ExternalCodeLoadDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  void pre(snap::Graph &graph,
           snap::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(snap::Graph &graph,
                snap::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(snap::Graph &graph,
            snap::program::Sequence &prog,
            poplar::DebugContext context) override;

  /**
   * Get the Function Buffer Mapping Type corresponding to the op's source /
   * destination types and locations, taken from the ExchangeDescriptor.
   *
   * \param 1 CodeMemoryType The destination code memory type to lookup.
   * \returns poplar::FunctionBufferMappingType
   */
  poplar::FunctionBufferMappingType
      getFunctionBufferMappingType(CodeMemoryType);
};

std::unique_ptr<ExchangeDescriptorx>
getExchangeDescriptorx(Devicex *dv_p, ExchangeDescriptor descriptor);

class ExchangeBaseOpx : public PopOpx {
public:
  ExchangeBaseOpx(Op *, Devicex *);
  std::set<TensorId> mustExistBeforeCreate(int) const override { return {}; }
};

} // namespace popx
} // namespace popart

#endif
