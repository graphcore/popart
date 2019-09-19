#include <popart/optimizervalue.hpp>

namespace popart {
bool OptimizerValue::validReplacement(const OptimizerValue &rhs) const {
  if (isConst() != rhs.isConst()) {
    return false;
  }
  if (isConst() && (val() - rhs.val() != 0.0f)) {
    return false;
  }
  return true;
}
} // namespace popart
