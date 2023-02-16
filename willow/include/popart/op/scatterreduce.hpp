// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SCATTERREDUCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SCATTERREDUCE_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"

namespace poprithms {
namespace ndarray {
class Shape;
} // namespace ndarray
} // namespace poprithms

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

enum class ScatterReduction { Sum = 0, Max, Min, Mul, None };

class ScatterReduceOp : public Op {
public:
  ScatterReduceOp(const OperatorIdentifier &_opid,
                  int64_t axis_,
                  int64_t axis_size_,
                  ScatterReduction reduction_,
                  int64_t group_size_,
                  bool enable_index_broadcast_,
                  const nonstd::optional<float> &available_memory_proportion_,
                  const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Which axis to scatter reduce on.
  int64_t getAxis() const { return axis; }
  int64_t getGroupSize() const { return group_size; }

  ScatterReduction getReduction() const { return reduction; }

  const Shape &getBackwardShape() const { return backward_shape; }

  static InIndex dataInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex initialValuesInIndex() { return 2; }
  static OutIndex outIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  nonstd::optional<float> getAvailableMemoryProportion() const {
    return available_memory_proportion;
  }

  void setAvailableMemoryProportion(const nonstd::optional<float> v) {
    available_memory_proportion = v;
  }

  static std::string reductionToString(ScatterReduction reduction);
  static ScatterReduction reductionFromString(const std::string &reductionStr);

  bool indexBroadcasted() const { return index_broadcasted; }
  bool indexBroadcastEnabled() const;

private:
  void setupOutputInfo();
  void checkIndexBroadcasted();

  Shape backward_shape;
  int64_t axis;
  int64_t axis_size;
  ScatterReduction reduction;
  int64_t group_size;
  nonstd::optional<float> available_memory_proportion;
  bool index_broadcasted;
  bool index_broadcast_enabled;
};

class ScatterReduceGradOp : public Op {
public:
  ScatterReduceGradOp(const ScatterReduceOp &op);

  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final {
    return mapper;
  }

  const std::map<int, int> &gradOutToNonGradIn() const final {
    return grad_out_info;
  }
  void setup() final;

  int64_t getAxis() const { return axis; }
  int64_t getGroupSize() const { return group_size; }

  ScatterReduction getReduction() const { return reduction; }

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex dataInIndex() { return 2; }
  static InIndex fwdOutInIndex() { return 3; }
  static InIndex initialValuesInIndex() { return 4; }

  static OutIndex gradDataOutIndex() { return 0; }
  static OutIndex gradInitialValuesOutIndex() { return 1; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  nonstd::optional<float> getAvailableMemoryProportion() const {
    return available_memory_proportion;
  }

  bool indexBroadcasted() const { return index_broadcasted; }
  bool indexBroadcastEnabled() const { return index_broadcast_enabled; }

  bool hasInitialValues() const { return has_initial_values; }

private:
  std::vector<GradInOutMapper> mapper;
  std::map<int, int> grad_out_info;
  Shape backward_shape;
  int64_t axis;
  ScatterReduction reduction;
  int64_t group_size;
  nonstd::optional<float> available_memory_proportion;
  bool index_broadcasted;
  bool index_broadcast_enabled;
  bool has_initial_values;
};

poprithms::ndarray::Shape
expandIndicesBcastNdShape(const poprithms::ndarray::Shape &indicesShape,
                          const poprithms::ndarray::Shape &dataShape,
                          unsigned int axis,
                          bool withGroups);
std::vector<std::size_t>
expandIndicesBcastShape(const std::vector<std::size_t> &indicesShape,
                        const std::vector<std::size_t> &dataShape,
                        unsigned int axis,
                        bool withGroups);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SCATTERREDUCE_HPP_
