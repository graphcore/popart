#ifndef GUARD_NEURALNET_POPOP_HPP
#define GUARD_NEURALNET_POPOP_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplar/DeviceManager.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#pragma clang diagnostic pop // stop ignoring warnings


namespace willow {

class PopOp{

  public:
    PopOp(Op *);
    virtual ~PopOp();

  // create the input poplar::Tensor for input at index
  virtual poplar::Tensor createInput(int index) = 0;
  virtual bool canCreateInput(int index) = 0;

  private:
  Op * op;

};


} // namespace willow

#endif
