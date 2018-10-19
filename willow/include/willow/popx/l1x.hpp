#ifndef GUARD_NEURALNET_L1X_HPP
#define GUARD_NEURALNET_L1X_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class L1GradOp;
class L1Op;

namespace popx {

class L1Opx : public Opx {
public:
  L1Opx(Op *);
  L1Op *getL1Op() const;
};

class L1GradOpx : public Opx {
public:
  L1GradOpx(Op *);
  L1GradOp *getL1GradOp() const;
};

} // namespace popx
} // namespace willow

#endif
