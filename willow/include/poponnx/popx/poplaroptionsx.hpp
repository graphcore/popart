#ifndef GUARD_NEURALNET_POPLAROPTIONSX_HPP
#define GUARD_NEURALNET_POPLAROPTIONSX_HPP

#include <iterator>
#include <map>
#include <string>

#include <poplar/OptionFlags.hpp>

namespace poponnx {
namespace popx {
// --- TODO ---
// THIS WHOLE FILE WILL BE REMOVED ONCE T5614 is done.
// We need the convolution options which are passed around to implement
// operand< in order for them to be used as a key for graph caching, however
// current poplar::OptionFlags does not do that, so we use our own structure
// instead and translate when needed.
struct PoplarOptions {
  std::map<std::string, std::string> options;

  /**
   * Converts poponnx Options to poplar OptionFlags.
   *
   * \return converted poplar OptionFlags
   */
  poplar::OptionFlags toOptionFlags() const {
    poplar::OptionFlags flags;
    for (auto &option : options) {
      flags.set(option.first, option.second);
    }
    return flags;
  }
};

} // namespace popx
} // namespace poponnx

#endif
