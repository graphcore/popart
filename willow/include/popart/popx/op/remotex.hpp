// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTEX_HPP
#define GUARD_NEURALNET_REMOTEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class RemoteBaseOpx : public Opx {
public:
  RemoteBaseOpx(Op *, Devicex *);

protected:
  poplar::Tensor makeWritable(poplar::Graph &sgraph,
                              poplar::Tensor t,
                              RemoteBufferId rbid,
                              TensorId id) const;
  void postLoad(poplar::program::Sequence &prog,
                RemoteBufferId rbid,
                const poplar::Tensor t) const;
  void preStore(poplar::Graph &sgraph,
                poplar::program::Sequence &prog,
                RemoteBufferId rbid,
                const poplar::Tensor t) const;
  void load(poplar::Graph &sgraph,
            poplar::program::Sequence &prog,
            RemoteBufferId rbid,
            const poplar::Tensor t,
            const poplar::Tensor offset) const;
  void store(poplar::program::Sequence &prog,
             RemoteBufferId rbid,
             const poplar::Tensor t,
             const poplar::Tensor offset) const;
};

class RemoteStoreOpx : public RemoteBaseOpx {
public:
  RemoteStoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class RemoteLoadOpx : public RemoteBaseOpx {
public:
  RemoteLoadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class RemoteExchangeOpx : public RemoteBaseOpx {
public:
  RemoteExchangeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  bool canUnwind(InIndex, OutIndex) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  poplar::Graph &inGraph(InIndex in) const;
  poplar::Graph &outGraph(OutIndex out) const;
};

} // namespace popx
} // namespace popart

#endif
