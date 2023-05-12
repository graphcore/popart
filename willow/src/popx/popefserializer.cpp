// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popart/popx/popefserializer.hpp"

#include <fstream>
#include <iostream>
#include <utility>

#include <poplar/Executable.hpp>

#include "popart/ir.hpp"
#include "popart/popx/devicex.hpp"
#include "popart/popx/executablex.hpp"
#include "popart/popx/irlowering.hpp"
#include "popart/vendored/optional.hpp"
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

namespace {
/** Performs a full sequential read of the .popef file.
 * To be called prior to loading by PopEF.
 */
void preloadFileIfEnvVarSet(const std::string &filepath) {
  const std::string preload_env_var_name = "PRELOAD_POPEF";
  const auto preload_env_var_value = getPopartEnvVar(preload_env_var_name);

  const std::string suffix = ".popef";
  auto endsWith            = [](const std::string &str,
                     const std::string &suffix) -> bool {
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
  };

  if (preload_env_var_value && *preload_env_var_value == "full-preload" &&
      // check it's a .popef file
      endsWith(filepath, suffix)) {
    try {
      logging::session::debug("Performing preload of popef file {}", filepath);
      std::ifstream input(filepath, std::ios::binary);
      std::ofstream output("/dev/null", std::ios::binary);
      if (!input) {
        throw std::runtime_error("Failed to open input file " + filepath +
                                 " for preload.");
      }
      if (!output) {
        throw std::runtime_error("Failed to open output stream '/dev/null'");
      }
      output << input.rdbuf();
      logging::session::debug("Completed preload of popef file {}", filepath);
    } catch (const std::exception &ex) {
      logging::session::warn("Error executing preload of file {}, where {} "
                             "env var has value {}. Error msg: {}",
                             filepath,
                             "POPART_" + preload_env_var_name,
                             *preload_env_var_value,
                             ex.what());
    }
  }
}
} // namespace

nonstd::optional<size_t>
Reader::checkFileForValidPoplarExecutable(const std::string &filePath) {
  auto ifs = std::make_shared<std::ifstream>(filePath, std::ifstream::binary);
  try {
    /** Preload (full sequential read of) the .popef file if the preload
     * environment variable is set. This is useful in storage environments where
     * the cached executables are far away, e.g. mounted s3 storage over a
     * network. In these cases, the first slow read over the network is much
     * faster by an explicit sequential read compared to via the Reader
     * instantiation below. After the first read, subsequent read speeds are
     * fast, and we can proceed with the Reader instantiation as normal.
     */
    preloadFileIfEnvVarSet(filePath);
    popart::popx::serialization::Reader reader({ifs});
    if (reader.containsExecutable() && reader.containsPoplarExecutable()) {
      auto hash = reader.readExecutableHash();
      logging::session::info("PopART cache file has been found: {}", filePath);
      return hash;
    } else {
      logging::session::info("Ignoring cache file because it does not contain "
                             "a valid PopART executable : {}",
                             filePath);
    }
  } catch (const std::exception &e) {
    logging::session::trace(
        "Ignoring invalid cache file {}: {}", filePath, e.what());
  }
  return {};
}

} // namespace serialization
} // namespace popx
} // namespace popart
