#ifndef GUARD_NEURALNET_AVERAGEPOOLX_HPP
#define GUARD_NEURALNET_AVERAGEPOOLX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class AveragePoolOp;
class AveragePoolGradOp;

namespace popx {

class AveragePoolOpx : public Opx {
public:
  AveragePoolOpx(Op *);
  AveragePoolOp *getAveragePoolOp() const;
};

class AveragePoolGradOpx : public Opx {
public:
  AveragePoolGradOpx(Op *);
  AveragePoolGradOp *getAveragePoolGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
