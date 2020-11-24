// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_NODEPATTERN_HPP
#define GUARD_NEURALNET_ONNXTOONNX_NODEPATTERN_HPP

#include <array>
#include <onnx/onnx_pb.h>
#include <string>

namespace popart {
namespace onnxpasses {

class Suffixer;

enum class ScalarInIndex { Zero, One };
class NodePattern {

private:
  ONNX_NAMESPACE::GraphProto &g;
  decltype(g.mutable_node()) nodes;

protected:
  using Node  = ONNX_NAMESPACE::NodeProto;
  using Graph = ONNX_NAMESPACE::GraphProto;

  Suffixer &suffixer;

public:
  NodePattern(ONNX_NAMESPACE::GraphProto &g_, Suffixer &suffixer_)
      : g(g_), nodes(g.mutable_node()), suffixer(suffixer_) {}

  virtual ~NodePattern() = default;

  /**
   * Execute this NodePattern on Node \a node.
   *
   *
   * \return true if this NodePattern matched \a node, and inserted
   *         replacement(s) into \a nodes. If true is returned, this Patterns
   *         hit count is incremented by 1.
   *
   * This method calls into the virtual method, \a go, which is the method which
   * NodePatterns which inherit from this base class should implement.
   * */
  bool run(const Node &node);

  /** The total number of Nodes which with NodePattern has found an match on. */
  int64_t nHits() const { return nHits_; }

protected:
  /** Insert a new empty Node into nodes */
  Node &blank() { return *nodes->Add(); }

  /** Insert a copy of the Node \a toCopy into nodes */
  Node &copy(const Node &toCopy);

  /** Insert a copy of \a toCopy into nodes, with modified inputs, outputs, and
   * type.
   *
   * \param toCopy The Node to copy (attributes etc will be taken from this).
   *
   * \param ins The 2 inputs to the Node, at input positions 0 and 1
   *            respectively.
   *
   * \param out The output of the Node, at output position 0.
   *
   * \param type The operator type of the Node, This string might be "Mul",
   *             "Div", etc
   *
   * */
  Node &binary(const Node &toCopy,
               const std::array<std::string, 2> &ins,
               const std::string &out,
               const std::string &type);

  /** Insert a copy of \a toCopy into nodes, with a modified input, output, and
   * operator type. */
  Node &unary(const Node &toCopy,
              const std::string &in_,
              const std::string &out,
              const std::string &type);

  /**
   * Set the input and output names of the Node, \a n */
  Node &setIO(Node &n,
              const std::vector<std::string> &inNames,
              const std::vector<std::string> &outNames);

  /**
   * Add to a constant scalar.
   * \sa binaryConstScalar
   * */
  Node &addConstScalar(const Node &toCopy,
                       const std::string &inName,
                       const std::string &outName,
                       ScalarInIndex inIndex,
                       float v) {
    return binaryConstScalar(toCopy, inName, outName, "Add", inIndex, v);
  }

  /**
   * Multiply by a constant scalar.
   * \sa binaryConstScalar
   * */
  Node &mulConstScalar(const Node &toCopy,
                       const std::string &inName,
                       const std::string &outName,
                       ScalarInIndex inIndex,
                       float v) {
    return binaryConstScalar(toCopy, inName, outName, "Mul", inIndex, v);
  }

  /**
   * Divide (or be divided by) a constant scalar.
   * \sa binaryConstScalar
   * */
  Node &divConstScalar(const Node &toCopy,
                       const std::string &inName,
                       const std::string &outName,
                       ScalarInIndex inIndex,
                       float v) {
    return binaryConstScalar(toCopy, inName, outName, "Div", inIndex, v);
  }

  /**
   * Subtract (or be subtracted from) a constant scalar.
   * \sa binaryConstScalar
   * */
  Node &subConstScalar(const Node &toCopy,
                       const std::string &inName,
                       const std::string &outName,
                       ScalarInIndex inIndex,
                       float v) {
    return binaryConstScalar(toCopy, inName, outName, "Sub", inIndex, v);
  }

  /**
   * Raise to the power of a constant scalar (appear as base or scalar).
   * \sa binaryConstScalar
   * */
  Node &powConstScalar(const Node &toCopy,
                       const std::string &inName,
                       const std::string &outName,
                       ScalarInIndex inIndex,
                       float v) {
    return binaryConstScalar(toCopy, inName, outName, "Pow", inIndex, v);
  }

private:
  /** Insert a copy of \a toCopy into nodes, but of operator type
   * "BinaryConstScalar".
   *
   * Recall that a BinaryConstScalar operator has 1 input and 1 output, where
   * the output is the result of applting a binary operation (Mul, Sub, etc) to
   * the input, with a scalar value, either on the left or right:
   *
   * \param toCopy The node to copy attributes, etc from.
   *
   * \param inName The input Tensor
   *
   * \param outName The output Tensor
   *
   * \param binaryOpType The binary operation. This can be one of Pow, Div, etc.
   *
   * \param inIndex The index at which the scalar will be the input to the
   *                 binary operation.
   *
   * \param v The value of the scalar. It will be cast to the type of the Tensor
   *          \a inName before the binary operation is performed.
   *
   * The method is particularly useful if the numerical type of the input Tensor
   * \a inName is not known when the transformation is applied.
   * */
  Node &binaryConstScalar(const Node &toCopy,
                          const std::string &inName,
                          const std::string &outName,
                          const std::string &binaryOpType,
                          ScalarInIndex inIndex,
                          float v);

  virtual bool go(const Node &node) = 0;
  int64_t nHits_{0};
};

} // namespace onnxpasses
} // namespace popart

#endif
