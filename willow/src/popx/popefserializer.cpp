// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popart/popx/popefserializer.hpp"

#include <iostream>
#include <utility>

#include <poplar/Executable.hpp>

#include "popart/ir.hpp"
#include "popart/popx/devicex.hpp"
#include "popart/popx/executablex.hpp"
#include "popart/popx/irlowering.hpp"
#include "popx/popefserializerimpl.hpp"

namespace popart {
namespace popx {
namespace serialization {

/** To see description go to the function declaration. */
Writer::Writer(std::ostream &out, const popart::popx::Devicex &device)
    : _impl(std::make_unique<WriterImpl>(out, device)) {}
Writer::Writer(Writer &&writer) : _impl(std::move(writer._impl)) {}
Writer::~Writer() = default;

/** To see description go to the function declaration. */
void Writer::serializePoplarExecutable() {
  _impl->serializePopefMetadata();
  _impl->serializePoplarEngine();
}

/** To see description go to the function declaration. */
void Writer::serializePopartMetadata() { _impl->serializePopartMetadata(); }

/** To see description go to the function declaration. */
void Writer::serializeTensorData() { _impl->serializeTensorData(); }

/** To see description go to the function declaration. */
Reader::Reader(const std::vector<std::shared_ptr<std::istream>> &in_vec)
    : _impl(std::make_unique<ReaderImpl>(in_vec)) {}
Reader::Reader(Reader &&reader) : _impl(std::move(reader._impl)) {}
Reader::~Reader() = default;

/** To see description go to the function declaration. */
size_t Reader::readExecutableHash() const { return _impl->_hash; }

/** To see description go to the function declaration. */
bool Reader::containsPoplarExecutable() const {
  return _impl->_poplarExecutable.has_value();
}

/** To see description go to the function declaration. */
bool Reader::containsExecutable() const {
  return _impl->_popartOpaque.has_value();
}

/** To see description go to the function declaration. */
bool Reader::containsPopefMetadata() {
  return _impl->_popefMetadata.has_value();
}

/** To see description go to the function declaration. */
poplar::Executable Reader::deserializePoplarExecutable() const {
  return _impl->deserializePoplarExecutable();
}

/** To see description go to the function declaration. */
std::unique_ptr<popart::popx::Executablex>
Reader::deserializeExecutable(popart::Ir &ir,
                              popart::popx::IrLowering &lowering) const {
  return _impl->deserializeExecutable(ir, lowering);
}

} // namespace serialization
} // namespace popx
} // namespace popart
