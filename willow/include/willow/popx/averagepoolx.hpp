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
  AveragePoolOpx(Op *, Devicex *);
  AveragePoolOp *getAveragePoolOp() const;
  void grow() const override final;
};

class AveragePoolGradOpx : public Opx {
public:
  AveragePoolGradOpx(Op *, Devicex *);
  AveragePoolGradOp *getAveragePoolGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
