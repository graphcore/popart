// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/optimizervalue.hpp>

#include <boost/functional/hash.hpp>

namespace popart {
namespace {
void assertNotEqual(const OptimizerValue &lhs, const OptimizerValue &rhs) {
  if (lhs == rhs) {
    throw internal_error(
        "Bug in OptimizerValue: `lhs.validReplacement(rhs)` has failed but "
        "`lhs == rhs`. These methods should be consistent. `lhs` = "
        "`{{isConst={}, val={}}}`. `rhs` = `{{isConst={}, val={}}}`.",
        lhs.isConst(),
        lhs.val(),
        rhs.isConst(),
        rhs.val());
  }
}
} // namespace

void OptimizerValue::validReplacement(const OptimizerValue &rhs) const {
  // To prevent future developer errors, on failure we assert that this is
  // consistent with operator==.

  if (isConst() != rhs.isConst()) {
    assertNotEqual(*this, rhs);
    throw error("Can not replace a constant value with a non constant value.");
  }
  if (isConst() && (val() - rhs.val() != 0.0f)) {
    assertNotEqual(*this, rhs);
    throw error("Values are constant but do not match.");
  }
}

bool OptimizerValue::operator==(const OptimizerValue &rhs) const {
  return (isConst() == rhs.isConst()) &&
         (!isConst() || (val() - rhs.val() == 0.0f));
}

} // namespace popart

namespace std {
std::size_t std::hash<popart::OptimizerValue>::
operator()(const popart::OptimizerValue &value) const {
  std::size_t seed = 0;
  if (value.isConst()) {
    boost::hash_combine(seed, value.val());
  }
  boost::hash_combine(seed, value.isConst());
  return seed;
}
} // namespace std
