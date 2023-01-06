// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_EXCHANGEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_EXCHANGEX_HPP_

#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/op/exchange/exchange.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorlocation.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

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

  void
  setInTensors(std::vector<std::pair<TensorId, poplar::Tensor>> inTensors_) {
    inTensors = inTensors_;
  }

  virtual void pre(poplar::Graph &graph,
                   poplar::program::Sequence &prog,
                   poplar::DebugContext context)      = 0;
  virtual void exchange(poplar::Graph &graph,
                        poplar::program::Sequence &prog,
                        poplar::DebugContext context) = 0;
  virtual void post(poplar::Graph &graph,
                    poplar::program::Sequence &prog,
                    poplar::DebugContext context)     = 0;

  virtual poplar::Tensor unwind(poplar::Graph &, poplar::Tensor) const;

  /**
   * Create an input tensor that is compatible with host and remote exchange
   * operations without causing rearrangements
   * \param graph Graph on which to create the tensor
   * \param info  Tensor info (data type and shape) of the tensor to create
   * \return      Tensor laid out optimally for exchanges with
   *              createHostTransferableTensor
   */
  virtual poplar::Tensor create(poplar::Graph &graph,
                                const TensorInfo &info) const;

  virtual std::vector<poplar::Tensor> getOutTensors() const {
    return outTensors;
  }

  virtual bool rearrangeOnHost() const { return true; }

protected:
  Devicex *dv_p;

  std::vector<std::pair<TensorId, poplar::Tensor>> inTensors;
  std::vector<poplar::Tensor> outTensors;

  ExchangeDescriptor descriptor;
};

class HostLoadDescriptorx : public ExchangeDescriptorx {
public:
  HostLoadDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  void pre(poplar::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(poplar::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(poplar::Graph &graph,
            poplar::program::Sequence &prog,
            poplar::DebugContext context) override;
  poplar::Tensor unwind(poplar::Graph &, poplar::Tensor) const override;
  bool rearrangeOnHost() const override;
};

class HostStoreDescriptorx : public ExchangeDescriptorx {
public:
  HostStoreDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  void pre(poplar::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(poplar::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(poplar::Graph &graph,
            poplar::program::Sequence &prog,
            poplar::DebugContext context) override;
  bool rearrangeOnHost() const override;
};

class RemoteLoadDescriptorx : public ExchangeDescriptorx {
public:
  RemoteLoadDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor_);
  void pre(poplar::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(poplar::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(poplar::Graph &graph,
            poplar::program::Sequence &prog,
            poplar::DebugContext context) override;
  poplar::Tensor unwind(poplar::Graph &, poplar::Tensor) const override;
};

class RemoteStoreDescriptorx : public ExchangeDescriptorx {
public:
  RemoteStoreDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  void pre(poplar::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(poplar::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(poplar::Graph &graph,
            poplar::program::Sequence &prog,
            poplar::DebugContext context) override;
};

class RemoteCodeLoadOpDescriptorx : public ExchangeDescriptorx {
public:
  RemoteCodeLoadOpDescriptorx(Devicex *dv_p_, ExchangeDescriptor descriptor);
  void pre(poplar::Graph &graph,
           poplar::program::Sequence &prog,
           poplar::DebugContext context) override;
  void exchange(poplar::Graph &graph,
                poplar::program::Sequence &prog,
                poplar::DebugContext context) override;
  void post(poplar::Graph &graph,
            poplar::program::Sequence &prog,
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

class ExchangeBaseOpx : public Opx {
public:
  ExchangeBaseOpx(Op *, Devicex *);
  std::set<TensorId> mustExistBeforeCreate(int) const override { return {}; }
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXCHANGE_EXCHANGEX_HPP_
