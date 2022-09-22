// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_CUSTOM_PARAMETERIZEDOP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_CUSTOM_PARAMETERIZEDOP_HPP_

#include <map>
#include <memory>
#include <utility>
#include <popart/graph.hpp> // IWYU pragma: keep
#include <popart/opmanager.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;
} // namespace popart

// make unique mask
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

namespace popart {

/**
 * @brief Generic base class for simple ops with parameterized attributes.
 *
 * The aim of this class is to regroup all the common logic in the
 * implementation of custom ops. In particular, it forces gathering all
 * parameters/attributes into a proper data structure, helping generalizing the
 * rest of the code.
 *
 * @tparam TDerivedOP CRTP template type.
 * @tparam TOpParams Structure containing the op parameters.
 */
template <typename TDerivedOp, typename TOpParams>
class ParameterizedOp : public popart::Op {
public:
  using ParamsType = TOpParams;

  /**
   * @brief Construct a custom op.
   * @param _opid Operator id (default one if not provided).
   * @param _params Operation parameters.
   * @param _settings Settings.
   */
  ParameterizedOp(const popart::OperatorIdentifier &_opid,
                  const ParamsType &_params,
                  const popart::Op::Settings &_settings)
      : popart::Op(_opid, _settings), m_params(_params) {}
  ParameterizedOp(const ParamsType &_params,
                  const popart::Op::Settings &_settings)
      : popart::Op(TDerivedOp::defaultOperatorId(), _settings),
        m_params(_params) {}

  /**
   * @brief Construct a custom op from another op with same parameters.
   * Typically, this constructor build a grad op from a fwd op.
   *
   * @tparam T Op input type.
   * @param _opid Operator identifier (default one if not provided).
   * @param _op Operation to extract setting and parameters from.
   */
  template <typename T>
  ParameterizedOp(const popart::OperatorIdentifier &_opid,
                  const ParameterizedOp<T, TOpParams> &_op)
      : popart::Op(_opid, _op.settings), m_params(_op.params()) {}
  template <typename T>
  ParameterizedOp(const ParameterizedOp<T, TOpParams> &_op)
      : popart::Op(TDerivedOp::defaultOperatorId(), _op.settings),
        m_params(_op.params()) {}

  /**
   * @brief Clone the operator. NOTE: using CRTP trick for generic
   * implementation!
   *
   * \return std::unique_ptr<Op> A unique pointer to the op.
   */
  std::unique_ptr<Op> clone() const override {
    return make_unique<TDerivedOp>(*dynamic_cast<const TDerivedOp *>(this));
  }

  /**
   * @brief Build the op from a PopART OpCreatorInfo data structure.
   *
   * @param info The OpCreatorInfo to use.
   *
   * @return Unique ptr of the op created.
   */
  static std::unique_ptr<TDerivedOp>
  createOpFromCreatorInfo(const popart::OpCreatorInfo &info) {
    auto params = TOpParams::makeFromAttributes(info.attributes);
    return make_unique<TDerivedOp>(info.opid, params, info.settings);
  }

  /**
   * @brief Create the custom op connected in a graph.
   *
   * @param graph Graph where to create and connect the op.
   * @param in Map of input tensor ids (i.e. name).
   * @param out Map of input tensor ids (i.e. name).
   * @param opid PopART operator identifier (default one if not provided).
   * @param params Custom op parameters.
   * @param settings Custom op settings.
   * @return Pointer to the custom op created (owned by the graph?)
   */
  static TDerivedOp *
  createOpInGraph(popart::Graph &graph,
                  const std::map<popart::InIndex, popart::TensorId> &in,
                  const std::map<popart::OutIndex, popart::TensorId> &out,
                  const popart::OperatorIdentifier &opid,
                  const TOpParams &params,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<TDerivedOp>(in, out, opid, params, settings);
  }
  static TDerivedOp *
  createOpInGraph(popart::Graph &graph,
                  const std::map<popart::InIndex, popart::TensorId> &in,
                  const std::map<popart::OutIndex, popart::TensorId> &out,
                  const TOpParams &params,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<TDerivedOp>(
        in, out, TDerivedOp::defaultOperatorId(), params, settings);
  }

  // Serialization of op attributes/parameters.
  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    m_params.appendAttributes(os);
  }
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    m_params.appendAttributes(os);
  }

  // PopART custom op additional config. Defaults.
  float getSubgraphValue() const override { return getHighSubgraphValue(); }
  bool requiresRandomSeed() const override { return false; }

  /**
   * @return Custom op parameters.
   */
  const TOpParams &params() const { return m_params; }

protected:
  ParamsType m_params;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_CUSTOM_PARAMETERIZEDOP_HPP_
