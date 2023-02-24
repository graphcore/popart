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

  virtual InIndex srcDataInIndex() const noexcept { return 0; }
  InIndex indicesInIndex() const noexcept { return 1; }
  virtual InIndex initialValuesInIndex() const noexcept { return 2; }
  OutIndex outIndex() const noexcept { return 0; }

  void setup() final;
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float getSubgraphValue() const override final;

  int64_t getAxis() const noexcept;
  int64_t getGroupSize() const noexcept;
  ScatterReduction getReduction() const noexcept;
  const Shape &getBackwardShape() const noexcept;
  bool indexBroadcasted() const noexcept;
  bool indexBroadcastEnabled() const noexcept;
  nonstd::optional<float> getAvailableMemoryProportion() const noexcept;
  void setAvailableMemoryProportion(const nonstd::optional<float> &v);

  static std::string reductionToString(ScatterReduction reduction);
  static ScatterReduction reductionFromString(const std::string &reductionStr);

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

  static InIndex gradInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex srcDataInIndex() { return 2; }
  static InIndex fwdOutInIndex() { return 3; }
  static InIndex initialValuesInIndex() { return 4; }
  static OutIndex gradDataOutIndex() { return 0; }
  static OutIndex gradInitialValuesOutIndex() { return 1; }

  void setup() override final;
  std::unique_ptr<Op> clone() const override final;
  const std::vector<GradInOutMapper> &gradInputInfo() const override final;
  const std::map<int, int> &gradOutToNonGradIn() const override final;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float getSubgraphValue() const override final;

  int64_t getAxis() const noexcept;
  int64_t getGroupSize() const noexcept;
  ScatterReduction getReduction() const noexcept;
  bool indexBroadcasted() const noexcept;
  bool indexBroadcastEnabled() const noexcept;
  bool hasInitialValues() const noexcept;
  nonstd::optional<float> getAvailableMemoryProportion() const noexcept;

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
