// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPERATORIDENTIFIER_HPP
#define GUARD_NEURALNET_OPERATORIDENTIFIER_HPP

#include <iosfwd>
#include <string>
#include <popart/names.hpp>

namespace popart {

namespace Domain {
constexpr const char *ai_onnx      = "ai.onnx";
constexpr const char *ai_onnx_ml   = "ai.onnx.ml";
constexpr const char *ai_graphcore = "ai.graphcore";
} // namespace Domain

struct NumInputs {

  int min;
  int max;

  NumInputs() : min(0), max(0) {}
  NumInputs(int f) : min(f), max(f) {}
  NumInputs(int _min, int _max) : min(_min), max(_max) {}
};

// The Op identifier is defined by ONNX a tuple
// (https://github.com/onnx/onnx/blob/master/docs/Versioning.md)
// domain.type:version
struct OperatorIdentifier {
  OperatorIdentifier(const OpDomain &_domain,
                     const OpType &_type,
                     OpVersion _version,
                     NumInputs inputs = {},
                     int outputs      = 0)
      : domain(_domain), type(_type), version(_version), numInputs(inputs),
        numOutputs(outputs) {

    // If no domain specified assume it is the default
    if (domain == "") {
      domain = Domain::ai_onnx;
    }
  }

  OpDomain domain;
  OpType type;
  OpVersion version;

  NumInputs numInputs;
  int numOutputs;

  bool operator==(const OperatorIdentifier &rhs) const {
    return (domain == rhs.domain && type == rhs.type && version == rhs.version);
  }

  bool operator!=(const OperatorIdentifier &rhs) const {
    return !(*this == rhs);
  }

  bool operator<(const OperatorIdentifier &rhs) const {
    return domain < rhs.domain || (domain == rhs.domain && type < rhs.type) ||
           (domain == rhs.domain && type == rhs.type && version < rhs.version);
  }
};

// The following does not work as we are in the popart namesapace >>
// template<>  struct less<OperatorIdentifier>
struct OperatorIdentifierLess {
  bool operator()(const OperatorIdentifier &lhs,
                  const OperatorIdentifier &rhs) const {
    if (lhs.domain < rhs.domain) {
      return true;
    } else if (lhs.domain > rhs.domain) {
      return false;
    } else {
      if (lhs.type < rhs.type) {
        return true;
      } else if (lhs.type > rhs.type) {
        return false;
      } else {
        if (lhs.version < rhs.version) {
          return true;
        } else {
          return false;
        }
      }
    }
  }
};

std::ostream &operator<<(std::ostream &os, const OperatorIdentifier &opid);

} // namespace popart

#endif
