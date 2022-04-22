// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IOCOMPUTETILECOPY_HPP
#define GUARD_NEURALNET_IOCOMPUTETILECOPY_HPP

#include <cstddef>
#include <string>
#include <popart/names.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/transforms/transform.hpp>

// IoComputeTileCopy:
// Transform that inserts IoTileCopy ops between the compute and IO graph of
// a single IPU.

namespace popart {
class Graph;
class Op;
class Tensor;

class IoComputeTileCopy : public Transform {
public:
  static std::size_t id();

  IoComputeTileCopy() : Transform() {}
  virtual ~IoComputeTileCopy() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "IoComputeTileCopy"; }

private:
  // Generate a name for the new tensor
  TensorId generateCopiedTensorId(Tensor *tensor, TileSet toIoTiles) const;

  void insertIoTileCopy(Graph &graph,
                        Tensor *tensor,
                        TileSet fromTileSet,
                        TileSet toTileSet,
                        Op *fromOp,
                        Op *toOp,
                        InIndex inIndex) const;

  void connectIoTileCopy(Graph &graph,
                         Tensor *tensor,
                         TileSet toTileSet,
                         Op *toOp,
                         InIndex inIndex) const;
};

} // namespace popart

#endif
