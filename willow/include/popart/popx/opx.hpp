// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OPX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OPX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <cstdint>
#include <snap/Tensor.hpp>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popart/popx/popopx.hpp>

#include "popart/names.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar
namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class Opx : public PopOpx {
public:
  // need to pass Devicex down to here, easy
  // access to poplar objects
  Opx(Op *, Devicex *);
  ~Opx();

  // create the input poplar::Tensor for input at index with name
  // default : throw error (not all PopOpxs can createInput)
  virtual poplar::Tensor createInput(InIndex index,
                                     const poplar::DebugNameAndId &dnai) const;
  snap::Tensor
  createInputTensor(popart::InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;

  // Hide Clang -Woverloaded-virtual on unwindTensorLayout by explicitly telling
  // compiler we want both PopOpx::unwindTensorLayout and the one defined below
  // (it thinks we may have made a typo and warns).
  using PopOpx::unwindTensorLayout;

  // Reverses the layout change to an input tensor for an op that returned
  // CANUNWIND
  virtual poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const;
  snap::Tensor
  unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const final;

  // If this Opx creates a poplar::Tensor at index0 (via createInput),
  // does it create the same poplar::Tensor as if opx1 creates one at
  // index1?. default behaviour : throws error
  virtual bool createsEquiv(int index0, const Opx *opx1, int index1) const;

  // clone the snap::Tensor identified by its TensorId, and copy the contents
  // of it.
  poplar::Tensor cloneNcopy(poplar::program::Sequence &, TensorId) const;
  // clone the snap::Tensor and copy the contents of it.
  poplar::Tensor cloneNcopy(poplar::program::Sequence &,
                            const poplar::Tensor &,
                            const std::string name = "") const;
  // Return the poplar Tensor identified by its TensorId, numpy broadcasting it
  // up to the given shape. Throws an exception if the identified Tensor doesn't
  // have a compatible shape.
  poplar::Tensor broadcast(const std::vector<int64_t> &, TensorId) const;
  // Return the given poplar Tensor, numpy broadcasting it up to the given
  // shape. Throws an exception if the given Tensor doesn't have a compatible
  // shape.
  poplar::Tensor broadcast(const std::vector<int64_t> &, poplar::Tensor) const;

  // Returns the virtual graph if enabled, else returns the dv_p->graph
  virtual poplar::Graph &graph() const;
  // shortcut for dv_p->tensors.get
  const poplar::Tensor &get(TensorId) const;
  // shortcut for dv_p->tensors.getView
  const poplar::Tensor &getView(TensorId) const;
  // shortcut for dv_p->tensors.insert
  void insert(TensorId, const poplar::Tensor &) const;

  // Return underlying Poplar input tensor
  const poplar::Tensor &getInTensor(InIndex index) const;

  // Return underlying Poplar output tensor
  const poplar::Tensor &getOutTensor(OutIndex index) const;

  // Return input tensor with shape matching IR specifications
  // (aliases getInTensor, but has any respective ViewChangers applied)
  const poplar::Tensor &getInView(InIndex index) const;

  // Return output tensor with shape matching IR specifications
  // (aliases getOutTensor, but has any respective ViewChangers applied)
  const poplar::Tensor &getOutView(OutIndex index) const;

  void setOutTensor(OutIndex index, const poplar::Tensor &tensor) const;

  // shortcut for dv_p->getConst
  poplar::Tensor getConst(const poplar::Type &type,
                          const std::vector<size_t> &shape,
                          double val,
                          const std::string &name) const;

  poplar::Tensor getScalarVariable(const poplar::Type &type,
                                   const std::string &name) const;

  using PopOpx::grow;
  void grow(snap::program::Sequence &) const final;
  virtual void grow(poplar::program::Sequence &) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OPX_HPP_
