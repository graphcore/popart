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
class Engine;
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

// Serialize both the poplar engine's executable, popart executable and the
// hash to the given ostream.
// poplarEngine / executable are optional and can be nullptr
void serializeEngineExecutable(std::ostream &out,
                               const poplar::Engine *poplarEngine,
                               const popart::popx::Executablex *executable,
                               size_t hash);

class Reader {
public:
  Reader(std::shared_ptr<std::istream> in);
  Reader(Reader &&reader);
  ~Reader();

  // Returns the executable hash or 0 if the stream contains
  // corrupted data
  size_t readExecutableHash() const;

  // Returns true if the stream contains a Poplar executable
  bool containsPoplarExecutable() const;

  // Returns true if the stream contains a Popart executable
  bool containsExecutable() const;

  // Returns true if the stream contains a Popef metadata
  bool containsPopefMetadata();

  // Load a poplar executable
  poplar::Executable deserializePoplarExecutable() const;

  // Load a popart executable
  std::unique_ptr<popart::popx::Executablex>
  deserializeExecutable(popart::Ir &ir,
                        popart::popx::IrLowering &lowering) const;

private:
  std::unique_ptr<ReaderImpl> _impl;
};

} // namespace serialization
} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_WILLOWEXECUTABLESERIALIZATION_HPP
