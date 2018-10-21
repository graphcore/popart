#ifndef GUARD_NEURALNET_PADX_HPP
#define GUARD_NEURALNET_PADX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

namespace willow {

class PadOp;

namespace popx {

class PadOpx : public Opx {
public:
  PadOpx(Op *, Devicex *);
  PadOp *getPadOp() const;
};

} // namespace popx
} // namespace willow

#endif
