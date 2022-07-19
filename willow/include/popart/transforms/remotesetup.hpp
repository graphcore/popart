// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_REMOTESETUP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_REMOTESETUP_HPP_

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <popart/transforms/transform.hpp>

#include "popart/names.hpp"

namespace popart {
struct POpIntCmp;
class Graph;
class Op;

using RemoteArgOpMap =
    std::map<TensorId, std::set<std::pair<Op *, InIndex>, POpIntCmp>>;
using RemoteOpArgMap =
    std::map<std::pair<Op *, InIndex>, std::set<TensorId>, POpIntCmp>;
using RemoteArgBufferMap =
    std::map<TensorId, std::pair<RemoteBufferId, RemoteBufferIndex>>;

class RemoteSetup : public Transform {
public:
  static std::size_t id();

  static void getRemoteArgMapping(Graph &graph,
                                  RemoteArgOpMap &,
                                  RemoteOpArgMap &,
                                  RemoteArgBufferMap &);

  RemoteSetup() : Transform() {}
  virtual ~RemoteSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "RemoteSetup"; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_REMOTESETUP_HPP_
