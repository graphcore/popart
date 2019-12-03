#ifndef GUARD_NEURALNET_BASESORT_HPP
#define GUARD_NEURALNET_BASESORT_HPP

#include <popart/op.hpp>

namespace popart {

class BaseSortOp : public Op {
public:
  BaseSortOp(const OperatorIdentifier &_opid,
             int64_t axis,
             const Op::Settings &settings);

  int64_t getAxis() const;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  static int getInIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

protected:
  // confirm that the axis is within the input tensor's rank
  void validateAxis() const;

private:
  const int64_t axis;
};

} // namespace popart

#endif
