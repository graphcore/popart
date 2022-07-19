// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_POPLAROPTIONSX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_POPLAROPTIONSX_HPP_

#include <map>
#include <string>
#include <utility>
#include <poplar/OptionFlags.hpp>

namespace popart {
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
   * Converts popart Options to poplar OptionFlags.
   *
   * \return Converted poplar OptionFlags.
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
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_POPLAROPTIONSX_HPP_
