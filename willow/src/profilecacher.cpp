// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "engineoptionscreator.hpp"
#include <boost/filesystem/operations.hpp>
#include <fileoperations.hpp>
#include <profilecacher.hpp>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/sessionoptions.hpp>

namespace popart {

ProfileCacher::ProfileCacher(const SessionOptions &opts,
                             const std::string &cachedExecutablePathStr,
                             const std::string &engineName)
    : cachedExecutableDir(
          boost::filesystem::path(cachedExecutablePathStr).parent_path()) {
  // Check if we can expect the profile.pop file
  bool expectProfile = false;
  const std::vector<std::string> optsCreatingProfile{
      "autoReport.all",
      "autoReport.outputGraphProfile",
      "autoReport.outputExecutionProfile"
      // ,
      // TODO: T63870 - The options below becomes important for the new API
      // "debug.instrument",
      // "debug.instrumentCompute",
      // "debug.instrumentControlFlow",
      // "debug.instrumentExternalExchange"
  };
  for (const auto &opt : optsCreatingProfile) {
    if (opts.engineOptions.find(opt) != opts.engineOptions.end()) {
      expectProfile = true;
      break;
    }
  }

  // Set the autoReport directory
  if (expectProfile) {
    if (opts.engineOptions.find("autoReport.directory") !=
        opts.engineOptions.end()) {
      autoReportDir = boost::filesystem::absolute(
          boost::filesystem::path(
              opts.engineOptions.at("autoReport.directory")) /
          engineName);
    } else {
      // Default path if no directory is set
      autoReportDir =
          boost::filesystem::absolute(boost::filesystem::path(engineName));
    }
  } else {
    // If we don't expect a profile, we will not search for them in the
    // autoReportDir
    autoReportDir = nonstd::nullopt;
  }
}

void ProfileCacher::storeProfilesToCache() const {
  if (!autoReportDir.has_value()) {
    // We need to specify "autoReport.directory" if we want to profile
    // If this is not set we do not profile (and would not know where to
    // copy the profiles from either)
    return;
  }

  logging::devicex::info("Storing profiles to cache");

  // Copy the profile.pop file
  auto profilePopSrcPath =
      findFileRecursively(autoReportDir.value(), "profile.pop");
  if (!boost::filesystem::exists(profilePopSrcPath)) {
    throw popart::error(
        logging::format("profile.pop not found in the autoReport dir: {}",
                        autoReportDir.value()));
  }
  auto profilePopDstPath = rebaseDirHierarchy(
      profilePopSrcPath,
      // parent_path() as we've added engineName to autoReportDir
      autoReportDir.value().parent_path(),
      cachedExecutableDir);
  boost::filesystem::create_directories(profilePopDstPath.parent_path());
  logging::devicex::debug(
      "Copying {} to {}", profilePopSrcPath, profilePopDstPath);
  boost::filesystem::copy_file(
      profilePopSrcPath,
      profilePopDstPath,
      boost::filesystem::copy_option::overwrite_if_exists);

  // Copy the debug.cbor file
  auto debugCborSrcPath =
      findFileRecursively(autoReportDir.value(), "debug.cbor");
  if (!boost::filesystem::exists(debugCborSrcPath)) {
    logging::devicex::info("debug.cbor not found in the autoReport dir: {}, "
                           "only proile.pop will be stored",
                           autoReportDir.value());
  } else {
    auto debugCborDstPath = rebaseDirHierarchy(
        debugCborSrcPath,
        // parent_path() as we've added engineName to autoReportDir
        autoReportDir.value().parent_path(),
        cachedExecutableDir);
    boost::filesystem::create_directories(debugCborDstPath.parent_path());
    logging::devicex::debug(
        "Copying {} to {}", debugCborSrcPath, debugCborDstPath);
    boost::filesystem::copy_file(
        debugCborSrcPath,
        debugCborDstPath,
        boost::filesystem::copy_option::overwrite_if_exists);
  }
}

void ProfileCacher::restoreProfilesFromCache() const {
  if (!autoReportDir.has_value()) {
    // We need to specify "autoReport.directory" if we want to profile
    // If this is not set we do not profile (and would not know where to
    // copy the profiles from either)
    return;
  }

  logging::devicex::info("Restoring profiles from cache");

  // Bail if the cached path has not been created
  if (!boost::filesystem::exists(cachedExecutableDir)) {
    logging::devicex::warn(
        "No profiles to restore as the cache dir {} does not exist",
        cachedExecutableDir);
    return;
  }

  // Copy the profile.pop file
  auto profilePopSrcPath =
      findFileRecursively(cachedExecutableDir, "profile.pop");
  if (!boost::filesystem::exists(profilePopSrcPath)) {
    logging::devicex::warn("profile.pop not found in the cache path: {}, "
                           "will not restore profiling information",
                           cachedExecutableDir);
    return;
  }
  auto profilePopDstPath = rebaseDirHierarchy(
      profilePopSrcPath,
      cachedExecutableDir,
      // parent_path() as we've added engineName to autoReportDir
      autoReportDir.value().parent_path());
  boost::filesystem::create_directories(profilePopDstPath.parent_path());
  logging::devicex::debug(
      "Copying {} to {}", profilePopSrcPath, profilePopDstPath);
  boost::filesystem::copy_file(
      profilePopSrcPath,
      profilePopDstPath,
      boost::filesystem::copy_option::overwrite_if_exists);

  // Copy the debug.cbor file
  auto debugCborSrcPath =
      findFileRecursively(cachedExecutableDir, "debug.cbor");
  if (!boost::filesystem::exists(debugCborSrcPath)) {
    logging::devicex::debug("debug.cbor not found in the cache path: {}, only "
                            "proile.pop will be restored",
                            cachedExecutableDir);
  } else {
    auto debugCborDstPath = rebaseDirHierarchy(
        debugCborSrcPath,
        cachedExecutableDir,
        // parent_path() as we've added engineName to autoReportDir
        autoReportDir.value().parent_path());
    boost::filesystem::create_directories(debugCborDstPath.parent_path());
    logging::devicex::debug(
        "Copying {} to {}", debugCborSrcPath, debugCborDstPath);
    boost::filesystem::copy_file(
        debugCborSrcPath,
        debugCborDstPath,
        boost::filesystem::copy_option::overwrite_if_exists);
  }
}

} // namespace popart
