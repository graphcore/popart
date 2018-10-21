#ifndef GUARD_NEURALNET_LOGSOFTMAXXXXX_HPP
#define GUARD_NEURALNET_LOGSOFTMAXXXXX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class LogSoftmaxOp;
class LogSoftmaxGradOp;

namespace popx {

class LogSoftmaxOpx : public Opx {
public:
  LogSoftmaxOpx(Op *, Devicex *);
  LogSoftmaxOp *getLogSoftmaxOp() const;
};

class LogSoftmaxGradOpx : public Opx {
public:
  LogSoftmaxGradOpx(Op *, Devicex *);
  LogSoftmaxGradOp *getLogSoftmaxGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
