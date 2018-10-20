#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class AddOp;
class AddGradOp;

namespace popx {

class AddOpx : public Opx {
public:
  AddOpx(Op *);
  AddOp *getAddOp() const;
};

class AddGradOpx : public Opx {
public:
  AddGradOpx(Op *);
  AddGradOp *getAddGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
