#ifndef GUARD_OPTION_FLAGS_HPP
#define GUARD_OPTION_FLAGS_HPP

#include <iterator>
#include <map>
#include <string>

namespace willow {

/**
 * A structure containing user configuration options for the Session class
 */
struct SessionOptions {
  /// Export 'dot' files of the forward and backward passes
  bool exportDot = false;

  /// Poplar engine options
  std::map<std::string, std::string> engineOptions;

  /// Poplar convolution options
  std::map<std::string, std::string> convolutionOptions;

  /// Poplar reporting options
  std::map<std::string, std::string> reportOptions;
};

} // namespace willow

#endif
