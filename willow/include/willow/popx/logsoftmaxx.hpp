#ifndef GUARD_NEURALNET_LOGSOFTMAXXXXX_HPP
#define GUARD_NEURALNET_LOGSOFTMAXXXXX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class LogSoftmaxOp;

namespace popx {

class LogSoftmaxOpx : public Opx {
public:
  LogSoftmaxOpx(Op *);
  LogSoftmaxOp *getLogSoftmaxOp() const;
};

} // namespace popx
} // namespace willow

#endif
