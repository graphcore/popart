// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPEXECUTABLESERIALIZATION_HPP
#define GUARD_NEURALNET_POPEXECUTABLESERIALIZATION_HPP

#include <map>
#include <memory>
#include <set>

#include <iostream>

namespace poplar {
// Forward declaration.
class Executable;
} // namespace poplar

namespace popart {

// Forward declaration.
class Ir;

namespace popx {

// Forward declaration.
class IrLowering;
class Executablex;

namespace serialization {

// Forward declaration.
class ReaderImpl;

// Serialize both the poplar executable, popart executable and the
// hash to the given ostream.
// poplarExecutable / executable are optional and can be nullptr
void serializeExecutable(std::ostream &out,
                         const poplar::Executable *poplarExecutable,
                         const popart::popx::Executablex *executable,
                         size_t hash);

class Reader {
public:
  Reader(const std::istream &in);
  ~Reader();

  // Returns the executable hash or 0 if the stream contains
  // corrupted data
  size_t readExecutableHash();

  // Return true if the stream contains a Poplar executable
  bool containsPoplarExecutable();

  // Return true if the stream contains a Popart executable
  bool containsExecutable();

  // Load a poplar executable
  poplar::Executable deserializePoplarExecutable();

  // Load a popart executable
  std::unique_ptr<popart::popx::Executablex>
  deserializeExecutable(popart::Ir &ir, popart::popx::IrLowering &lowering);

private:
  std::unique_ptr<ReaderImpl> _impl;
};

} // namespace serialization
} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_WILLOWEXECUTABLESERIALIZATION_HPP
