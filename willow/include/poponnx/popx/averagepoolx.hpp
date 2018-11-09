#ifndef GUARD_NEURALNET_AVERAGEPOOLX_HPP
#define GUARD_NEURALNET_AVERAGEPOOLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

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
  void grow() const override final;
};

} // namespace popx
} // namespace willow

#endif
