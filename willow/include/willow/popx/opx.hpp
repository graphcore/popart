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
  virtual bool canCreateInput(int index) const;
  // If this Opx creates a poplar::Tensor at index0 (via createInput),
  // does it create the same poplar::Tensor as if opx1 creates one at
  // index1?. default behaviour : throws error
  virtual bool createsEquiv(int index0, Opx *opx1, int index1) const;

  Op *getOp() const;
  Devicex *getDevx() const;

private:
  Op *op_;
  Devicex *devicex_;
};

} // namespace popx
} // namespace willow

#endif
