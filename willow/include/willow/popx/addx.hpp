#ifndef GUARD_NEURALNET_ADDX_HPP
#define GUARD_NEURALNET_ADDX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class AddOp;

namespace popx {

class AddOpx : public Opx {
public:
  AddOpx(Op *);
  AddOp *getAddOp() const;
};

} // namespace popx
} // namespace willow

#endif
