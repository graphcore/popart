// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTEX_HPP
#define GUARD_NEURALNET_REMOTEX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class RemoteBaseOpx : public PopOpx {
public:
  RemoteBaseOpx(Op *, Devicex *);

protected:
  poplar::Tensor makeWritable(snap::Graph &sgraph,
                              poplar::Tensor t,
                              RemoteBufferId rbid,
                              TensorId id) const;
  void postLoad(poplar::program::Sequence &prog,
                RemoteBufferId rbid,
                const poplar::Tensor t) const;
  void preStore(snap::Graph &sgraph,
                poplar::program::Sequence &prog,
                RemoteBufferId rbid,
                const poplar::Tensor t) const;
  void load(snap::Graph &sgraph,
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
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class RemoteExchangeOpx : public RemoteBaseOpx {
public:
  RemoteExchangeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  bool canUnwind(InIndex, OutIndex) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  snap::Graph &inGraph(InIndex in) const;
  snap::Graph &outGraph(OutIndex out) const;
};

} // namespace popx
} // namespace popart

#endif
