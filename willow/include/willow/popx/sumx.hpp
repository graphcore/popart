#ifndef GUARD_NEURALNET_SUMX_HPP
#define GUARD_NEURALNET_SUMX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class SumOp;

namespace popx {

class SumOpx : public Opx {
public:
  SumOpx(Op *);
  SumOp *getSumOp() const;
};

} // namespace popx
} // namespace willow

#endif
