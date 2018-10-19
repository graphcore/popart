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

class Opx {

public:
  Opx(Op *);
  virtual ~Opx();

  // create the input poplar::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  virtual poplar::Tensor createInput(int index) const;
  // default return false
  virtual bool canCreateInput(int index) const;

  Op *getOp() const;

private:
  Op *op;
};

} // namespace popx
} // namespace willow

#endif
