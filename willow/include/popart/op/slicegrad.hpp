#ifndef GUARD_NEURALNET_SLICEGRAD_HPP
#define GUARD_NEURALNET_SLICEGRAD_HPP

#include <popart/op/pad.hpp>

// The SliceGradOp has been moved out of the slice.hpp due to a
// circular dependency between PadGradOp inheriting from SliceOp
// and SliceGradOp inheriting from PadOp

namespace popart {

class SliceOp;

class SliceGradOp : public PadOp {
public:
  SliceGradOp(const SliceOp &);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  static std::vector<int64_t> calculatePadding(const SliceOp &);
};

} // namespace popart

#endif
