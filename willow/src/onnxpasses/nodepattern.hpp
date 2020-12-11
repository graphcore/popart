// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_NODEPATTERN_HPP
#define GUARD_NEURALNET_ONNXTOONNX_NODEPATTERN_HPP

#include <array>
#include <onnx/onnx_pb.h>
#include <onnxpasses/onnxnames.hpp>
#include <string>
#include <poprithms/ndarray/shape.hpp>

namespace popart {
namespace onnxpasses {

class PatternTarget;

enum class ScalarInIndex { Zero, One };

/**
 * An abstract base pass for performing transformations on ONNX Graphs.
 * */
class NodePattern {

protected:
  std::string withUniqueSuffix(const std::string &) const;

  // Get the Shape of a Tensor, retrieved from the target's GraphProto.
  poprithms::ndarray::Shape shape(const std::string &name) const;

private:
  // Multiple NodePatterns modify a single GraphProto, and use shared
  // resources. One such shared resource is a Suffixer object, used for
  // generating unique Tensor names. These shared resources are encapsulated in
  // a PatternTarget object:
  std::shared_ptr<PatternTarget> target;

  // Insert an empty NodeProto into the Graph.
  NodeProto &add();

public:
  NodePattern(std::shared_ptr<PatternTarget> t) : target(t) {}

  virtual ~NodePattern() = default;

  /**
   * Execute this NodePattern on NodeProto \a node.
   *
   * \return true if this NodePattern matched \a node, and inserted
   *         replacement(s) into \a nodes. If true is returned, this Pattern's
   *         hit count is incremented by 1.
   *
   * This method calls into the virtual method, \a go, which is the method which
   * NodePatterns inheriting from this base class should implement.
   * */
  bool run(const NodeProto &node);

  /** The total number of Nodes which with NodePattern has found an match on. */
  int64_t nHits() const { return nHits_; }

protected:
  /** Insert a NodeProto with no inputs and no outputs, and which has only the
   * attributes of \a src which are prefixed with "__". These attributes are
   * generally used for IPU specific things like pipeline stage, IPU number,
   * etc. */
  NodeProto &copyUnderscorePrefixedAttributes(const NodeProto &src);

  /** Insert a copy of the NodeProto \a toCopy into nodes. This is an exact
   * copy. */
  NodeProto &copy(const NodeProto &toCopy);

  /** Insert a copy of \a toCopy into nodes, with modified inputs, outputs, and
   * type. Attributes are copied exactly.
   *
   * \param toCopy The NodeProto to copy (attributes etc. will be taken from
   * this).
   *
   * \param ins The 2 inputs to the Node, at input indices 0 and 1
   *            respectively.
   *
   * \param out The output of the Node, at output index 0.
   *
   * \param type The operator type of the Node, This string might be "Mul",
   *             "Div", etc
   *
   * */
  NodeProto &binary(const NodeProto &toCopy,
                    const std::array<std::string, 2> &ins,
                    const std::string &out,
                    const std::string &type);

  /** Insert a copy of \a toCopy into nodes, with a modified input, output, and
   * operator type. \sa binary */
  NodeProto &unary(const NodeProto &toCopy,
                   const std::string &in_,
                   const std::string &out,
                   const std::string &type);

  /**
   * Set the input and output names of the Node, \a n */
  NodeProto &setIO(NodeProto &n,
                   const std::vector<std::string> &inNames,
                   const std::vector<std::string> &outNames);

  /**
   * Add to a constant scalar to the Tensor with nome \a inName.
   * \sa binaryConstScalar
   * */
  NodeProto &addConstScalar(const NodeProto &toCopy,
                            const std::string &inName,
                            const std::string &outName,
                            ScalarInIndex inIndex,
                            float v) {
    return binaryConstScalar(toCopy, inName, outName, "Add", inIndex, v);
  }

  /**
   * Multiply the Tensor with name \a inName by a constant scalar.
   * \sa binaryConstScalar
   * */
  NodeProto &mulConstScalar(const NodeProto &toCopy,
                            const std::string &inName,
                            const std::string &outName,
                            ScalarInIndex inIndex,
                            float v) {
    return binaryConstScalar(toCopy, inName, outName, "Mul", inIndex, v);
  }

  /**
   * Divide the Tensor with name \a inName a constant scalar, or divide a
   * constant scalar by this Tensor. \sa binaryConstScalar
   * */
  NodeProto &divConstScalar(const NodeProto &toCopy,
                            const std::string &inName,
                            const std::string &outName,
                            ScalarInIndex inIndex,
                            float v) {
    return binaryConstScalar(toCopy, inName, outName, "Div", inIndex, v);
  }

  /**
   * Subtract the Tensor with name \a inName from a constant scalar, or subtract
   * a constant scalar from it. \sa binaryConstScalar
   * */
  NodeProto &subConstScalar(const NodeProto &toCopy,
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
  NodeProto &powConstScalar(const NodeProto &toCopy,
                            const std::string &inName,
                            const std::string &outName,
                            ScalarInIndex inIndex,
                            float v) {
    return binaryConstScalar(toCopy, inName, outName, "Pow", inIndex, v);
  }

private:
  /** Insert a copy of \a toCopy into nodes, but change the operator type to
   * "BinaryConstScalar".
   *
   * Recall that a BinaryConstScalar operator has 1 input and 1 output, where
   * the output is the result of applting a binary operation (Mul, Sub, etc) to
   * the input, with a scalar value, either on the left or right:
   *
   * \param toCopy The node to copy attributes, etc. from.
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
  NodeProto &binaryConstScalar(const NodeProto &toCopy,
                               const std::string &inName,
                               const std::string &outName,
                               const std::string &binaryOpType,
                               ScalarInIndex inIndex,
                               float v);

  virtual bool go(const NodeProto &node) = 0;
  int64_t nHits_{0};
};

} // namespace onnxpasses
} // namespace popart

#endif
