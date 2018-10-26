#ifndef GUARD_NEURALNET_POPOP_HPP
#define GUARD_NEURALNET_POPOP_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

#include <willow/names.hpp>

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


  // The Op corresponding to this Opx
  Op * op_p;

  // The Devicex to which this Opx belongs
  Devicex * dv_p;

};

} // namespace popx
} // namespace willow

#endif
