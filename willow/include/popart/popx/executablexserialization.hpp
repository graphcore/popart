// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPEXECUTABLESERIALIZATION_HPP
#define GUARD_NEURALNET_POPEXECUTABLESERIALIZATION_HPP

#include <map>
#include <memory>
#include <set>

#include <iostream>
#include <popart/popx/irlowering.hpp>

namespace popart {
namespace popx {
namespace serialization {

// Serialize both the poplar executable, popart executable and the
// hash to the given ostream.
// poplarExecutable / executable are optional and can be nullptr
void serializeExecutable(std::ostream &out,
                         const poplar::Executable *poplarExecutable,
                         const popart::popx::Executablex *executable,
                         size_t hash);

// Returns the executable hash or 0 if the stream doesn't point at a valid
// serialized Header.
// The input stream must be pointing at the beginning of a Header
// Note: this function will not change the stream position
size_t readExecutableHash(std::istream &in);

// Return true if the stream contains a Poplar executable
// The input stream must be pointing at the beginning of a Header
// Note: this function will not change the stream position
bool containsPoplarExecutable(std::istream &in);

// Return true if the stream contains a Popart executable
// The input stream must be pointing at the beginning of a Header
// Note: this function will not change the stream position
bool containsExecutable(std::istream &in);

// Move the given stream to the end of the data.
// The input stream must be pointing at the beginning of a Header
void moveStreamToEnd(std::istream &in);

// Load a popart executable from the given stream.
//
// The input stream must be pointing at the beginning of a Header
// Note: this function will not change the stream position
std::unique_ptr<popart::popx::Executablex>
deserializeExecutable(std::istream &in,
                      popart::Ir &ir,
                      popart::popx::IrLowering &lowering);

// Load a popart executable from the given stream.
//
// The input stream must be pointing at the beginning of a Header
// Note: this function will not change the stream position
poplar::Executable deserializePoplarExecutable(std::istream &in);

} // namespace serialization
} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_WILLOWEXECUTABLESERIALIZATION_HPP
