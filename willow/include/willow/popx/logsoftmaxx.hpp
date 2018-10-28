#ifndef GUARD_NEURALNET_LOGSOFTMAXXXXX_HPP
#define GUARD_NEURALNET_LOGSOFTMAXXXXX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class LogSoftmaxOp;
class LogSoftmaxGradOp;
class LogSoftmaxGradDirectOp;

namespace popx {

class LogSoftmaxOpx : public Opx {
public:
  LogSoftmaxOpx(Op *, Devicex *);
  LogSoftmaxOp *getLogSoftmaxOp() const;
  void grow() const override final;
};

class LogSoftmaxGradOpx : public Opx {
public:
  LogSoftmaxGradOpx(Op *, Devicex *);
  LogSoftmaxGradOp *getLogSoftmaxGradOp() const;
};

class LogSoftmaxGradDirectOpx : public Opx {
public:
  LogSoftmaxGradDirectOpx(Op *, Devicex *);
  LogSoftmaxGradDirectOp *getLogSoftmaxGradDirectOp() const;
};

} // namespace popx
} // namespace willow

#endif
