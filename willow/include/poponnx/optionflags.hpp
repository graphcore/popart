#ifndef GUARD_OPTION_FLAGS_HPP
#define GUARD_OPTION_FLAGS_HPP

#include <iterator>
#include <map>
#include <string>

namespace poponnx {

/**
 * A structure containing user configuration options for the Session class
 */
struct SessionOptions {

  SessionOptions &operator=(const SessionOptions &rhs) = default;

  /// A directory for log traces to be written into
  std::string logDir;

  /// Export 'dot' files of the forward and backward passes
  bool exportDot = false;

  /// Controls caching of the convolution graphs. If set to false, then none of
  ///  the convolutions will be cached.
  bool enableConvolutionGraphCaching = true;

  /// Enable recomputation of marked operations in the graph
  bool enableRecomputation = false;

  /// Enable placement of operations on individual IPUs by creating a 'virtual
  /// graph' for each IPU
  bool enableVirtualGraphs = false;

  /// Use synthetic data i.e. disable data transfer to/from the host
  /// Set to 'true' to use synthetic data, 'false' to use real data
  bool ignoreData = false;

  /// Poplar engine options
  std::map<std::string, std::string> engineOptions;

  /// Poplar convolution options
  std::map<std::string, std::string> convolutionOptions;

  /// Poplar reporting options
  std::map<std::string, std::string> reportOptions;

  /// Logging options for poponnx
  std::map<std::string, std::string> loggingOptions;
};

} // namespace poponnx

#endif
