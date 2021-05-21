// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPOP_HPP
#define GUARD_NEURALNET_POPOP_HPP

#include <poplar/Graph.hpp>

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class Opx : public PopOpx {
public:
  // need to pass Devicex down to here, easy
  // access to poplar objects
  Opx(Op *, Devicex *);
  ~Opx();

  // If this Opx creates a poplar::Tensor at index0 (via createInput),
  // does it create the same poplar::Tensor as if opx1 creates one at
  // index1?. default behaviour : throws error
  virtual bool createsEquiv(int index0, const Opx *opx1, int index1) const;

  // Returns the virtual graph if enabled, else returns the dv_p->graph
  virtual poplar::Graph &graph() const;
  // The default assumes all Opx input and output tensors are laid out on the
  // same virtual graph. These methods should be overridden when this is not
  // the case, such as for IpuCopyOpx.
  // Returns the virtual graph for the tensor at InIndex, defaults to graph()
  virtual poplar::Graph &srcGraph(InIndex) const;
  // Returns the virtual graph for the tensor at OutIndex, defaults to graph()
  virtual poplar::Graph &dstGraph(OutIndex) const;
};

} // namespace popx
} // namespace popart

#endif
