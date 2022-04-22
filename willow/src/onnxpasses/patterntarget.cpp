// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cstdint>
#include <onnxpasses/patterntarget.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/printiter.hpp>
#include <popart/error.hpp>

#include "popart/logging.hpp"

namespace popart {
namespace onnxpasses {

PatternTarget::PatternTarget(GraphProto &g_)
    : g(g_), nodes(g.mutable_node()), suffixer(g_) {
  extractShapes();
  foldConstants = std::make_shared<Constants>();
}

void PatternTarget::appendShapes(ValueInfoProtos &valueInfoProtos) {

  for (const auto &valueInfoProto : valueInfoProtos) {
    if (valueInfoProto.has_name()) {
      const auto &tensorId = valueInfoProto.name();
      if (valueInfoProto.has_type()) {
        const auto &type = valueInfoProto.type();
        if (type.has_tensor_type()) {
          const auto &tensorType = type.tensor_type();
          if (tensorType.has_shape()) {
            const auto &shape = tensorType.shape();
            const auto &dims  = shape.dim();
            // Dimensions can either be strings or values. We ignore the case
            // where it is a string.
            if (std::any_of(dims.cbegin(), dims.cend(), [](const auto &x) {
                  return !x.has_dim_value();
                })) {
              continue;
            }

            std::vector<int64_t> shape_;
            shape_.reserve(dims.size());
            for (const auto &dim : dims) {
              shape_.push_back(dim.dim_value());
            }
            const auto pShape = poprithms::ndarray::Shape(shape_);
            const auto found  = shapes.find(tensorId);
            if (found != shapes.cend()) {
              if (found->second != pShape) {
                std::ostringstream oss;
                oss << "Failure in PatternTarget constructor, "
                    << "where the Tensor with name " << tensorId
                    << " has at least 2 shapes registered in the Graph, "
                    << "with different Shapes registered:" << found->second
                    << " != " << pShape << '.';
                throw error(oss.str());
              }
            } else {
              shapes.insert({tensorId, pShape});
            }
          }
        }
      }
    }
  }
}

void PatternTarget::extractShapes() {
  appendShapes(g.input());
  appendShapes(g.value_info());
  appendShapes(g.output());
}

poprithms::ndarray::Shape PatternTarget::shape(const std::string &id) const {
  const auto found = shapes.find(id);
  if (found == shapes.cend()) {
    std::ostringstream oss;
    oss << "\nFailed in PatternTarget::shape(TensorId = " << id << "), "
        << "as there is no Shape in this PatternTarget's map registered for `"
        << id << "'. There are Shape registered for " << shapes.size()
        << " TensorIds. ";

    std::vector<std::string> nodesInferred;
    std::vector<std::string> nodesNotInferred;
    for (const auto &n : g.node()) {
      if (std::any_of(n.output().cbegin(),
                      n.output().cend(),
                      [this](const auto &x) { return shapes.count(x) == 0; })) {
        nodesNotInferred.push_back(n.op_type());
      } else {
        nodesInferred.push_back(n.op_type());
      }
    }
    oss << "\nNodes with all outputs inferred: ";
    poprithms::util::append(oss, nodesInferred);
    oss << "\nNodes with NOT all outputs inferred: ";
    poprithms::util::append(oss, nodesNotInferred);
    oss << ". ";
    throw error(oss.str());
  }
  return found->second;
}

} // namespace onnxpasses
} // namespace popart
