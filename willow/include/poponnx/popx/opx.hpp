#ifndef GUARD_NEURALNET_POPOP_HPP
#define GUARD_NEURALNET_POPOP_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

#include <poponnx/names.hpp>

namespace willow {

namespace popx {

class Devicex;

class Opx {

public:
  // need to pass Devicex down to here, easy
  // access to poplar objects
  Opx(Op *, Devicex *);
  virtual ~Opx();

  // create the input poplar::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  virtual poplar::Tensor createInput(int index) const;
  // default return false
  virtual bool canCreateInput(int index0) const;
  // If this Opx creates a poplar::Tensor at index0 (via createInput),
  // does it create the same poplar::Tensor as if opx1 creates one at
  // index1?. default behaviour : throws error
  virtual bool createsEquiv(int index0, Opx *opx1, int index1) const;
  // To create a poplar::Tensor for input index index0, which
  // poplar::Tensors must already exist?
  virtual std::vector<TensorId> mustExistBeforeCreate(int index0) const;
  // adds poplar::Tensors to devicex_->popTensors,
  // one for each output of op_.
  virtual void grow() const;
  // clone the poplar::Tensor identified by its TensorId,
  // and copy the contents of it, in the step() program.
  poplar::Tensor cloneNcopy(TensorId) const;

  // The following reduce boilerplate.

  // shortcut for dv_p->graph
  poplar::Graph &graph() const;
  // shortcut for dv_p->tensors.get
  const poplar::Tensor &get(TensorId) const;
  // shortcut for dv_p->tensors.insert
  void insert(TensorId, const poplar::Tensor &) const;
  // shortcut for dv_p->progs.step
  poplar::program::Sequence &step() const;
  // shortcut for op_p->input.id(int)
  TensorId inId(int) const;
  // shortcut for op_p->output.id(int)
  TensorId outId(int) const;
  // shortcut for std::to_string(op_p->id)
  std::string idStr() const;

  // The Op corresponding to this Opx
  Op *op_p;

  // The Devicex to which this Opx belongs
  Devicex *dv_p;
};

} // namespace popx
} // namespace willow

#endif
