// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_PROFILECACHER_HPP_
#define POPART_WILLOW_SRC_PROFILECACHER_HPP_

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <string>
#include <popart/vendored/optional.hpp>

namespace popart {

struct SessionOptions;

/**
 * \brief Class for storing and restoring profiling files to and from the cache
 * dir.
 *
 * \warning In order to profile cached executables the user must specify
 *          ``autoReport`` options through ``SessionOptions``.
 *          Specification through the environmental variable
 *          ``POPLAR_ENGINE_OPTIONS`` will not work.
 */
class ProfileCacher {
public:
  /**
   * \brief Construct a new Cached Executable Profiler object
   *
   * \param opts The session options
   * \param cachedExecutablePathStr The path to the cached executable
   * \param engineName Name of the engine
   */
  // TODO: T63870 - Parse the equivalent of POPLAR_ENGINE_OPTIONS in addition to
  // opts when ready
  ProfileCacher(const SessionOptions &opts,
                const std::string &cachedExecutablePathStr,
                const std::string &engineName);

  /// Store profile files to cache directory if autoReportDirSet is true.
  void storeProfilesToCache() const;
  /// Restore profile files from cache directory if autoReportDirSet is true.
  void restoreProfilesFromCache() const;

private:
  /// Directory of the cached executable directory
  const boost::filesystem::path cachedExecutableDir;
  /// Directory pointed to by "autoReport.directory" (if set)
  nonstd::optional<boost::filesystem::path> autoReportDir;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_PROFILECACHER_HPP_
