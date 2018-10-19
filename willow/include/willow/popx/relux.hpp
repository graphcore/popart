#ifndef GUARD_NEURALNET_RELUX_HPP
#define GUARD_NEURALNET_RELUX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class ReluOp;

namespace popx {

class ReluOpx : public Opx {
public:
  ReluOpx(Op *);
  ReluOp *getReluOp() const;
};

} // namespace popx
} // namespace willow

#endif
