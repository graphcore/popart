#ifndef GUARD_NEURALNET_AVERAGEPOOLX_HPP
#define GUARD_NEURALNET_AVERAGEPOOLX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class AveragePoolOp;

namespace popx {

class AveragePoolOpx : public Opx {
public:
  AveragePoolOpx(Op *);
  AveragePoolOp *getAveragePoolOp() const;
};

} // namespace popx
} // namespace willow

#endif
