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
  LogSoftmaxOpx(Op *);
  LogSoftmaxOp *getLogSoftmaxOp() const;
};

class LogSoftmaxGradOpx : public Opx {
public:
  LogSoftmaxGradOpx(Op *);
  LogSoftmaxGradOp *getLogSoftmaxGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
