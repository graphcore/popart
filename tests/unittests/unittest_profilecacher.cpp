// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ProfileCacher

#include "profilecacher.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>
#include <fileoperations.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/sessionoptions.hpp>

struct ProfileCacherFixture {
  ProfileCacherFixture()
      : tmpDirRoot(boost::filesystem::temp_directory_path() /
                   boost::filesystem::unique_path()) {
    // Create tmp directory
    boost::filesystem::create_directories(tmpDirRoot);
  }

  // Function for setting up the environment so that ProfileCacher can be tested
  // in isolation
  popart::ProfileCacher
  getProfileCacher(const std::string &engineName     = "inference",
                   const std::string &creationOption = "",
                   std::string autoReportDirStr      = "") {
    // We set the current working directory to the tmpDir to ensure that all
    // files are removed on destruction
    boost::filesystem::current_path(tmpDirRoot);

    // Set the autoReportDir and session options
    auto autoReportDir = tmpDirRoot;
    if (!autoReportDirStr.empty()) {
      opts.engineOptions["autoReport.directory"] = autoReportDirStr;
      autoReportDir /= boost::filesystem::path(autoReportDirStr);
    }
    if (!creationOption.empty()) {
      opts.engineOptions[creationOption] = "true";
    }
    // Engine name is always appended to the dir
    autoReportDir /= engineName;

    // Set the file paths
    cachedExecutablePathStr =
        tmpDirRoot.string() + "/savedExecutable/model.popef";
    auto cachedExecutableDir =
        boost::filesystem::path(cachedExecutablePathStr).parent_path();

    profileSrcPath = autoReportDir / "profile.pop";
    debugSrcPath   = autoReportDir / "debug.cbor";
    profileDstPath = cachedExecutableDir / engineName / "profile.pop";
    debugDstPath   = cachedExecutableDir / engineName / "debug.cbor";

    // Create directories
    boost::filesystem::create_directories(cachedExecutableDir);
    boost::filesystem::create_directories(autoReportDir);

    // Create the (empty) files
    if (!creationOption.empty()) {
      std::ofstream profileStream(profileSrcPath.string());
      std::ofstream debugStream(debugSrcPath.string());
    }

    // Ensure that the setup is correct
    BOOST_ASSERT(boost::filesystem::exists(autoReportDir));
    if (!creationOption.empty()) {
      BOOST_ASSERT(boost::filesystem::exists(profileSrcPath));
      BOOST_ASSERT(boost::filesystem::exists(debugSrcPath));
    } else {
      BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
      BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));
    }
    BOOST_ASSERT(!boost::filesystem::exists(profileDstPath));
    BOOST_ASSERT(!boost::filesystem::exists(debugDstPath));

    return popart::ProfileCacher(opts, cachedExecutablePathStr, engineName);
  }

  ~ProfileCacherFixture() {
    // Clean-up
    boost::filesystem::remove_all(tmpDirRoot);
  }

  bool checkErrorMsg(const popart::error &ex) {
    const auto expectedPrefix = "profile.pop not found in the autoReport dir:";
    return boost::algorithm::starts_with(ex.what(), expectedPrefix);
  }

  boost::filesystem::path tmpDirRoot;

  // Fixture parameters
  popart::SessionOptions opts;
  std::string cachedExecutablePathStr;

  // File paths
  boost::filesystem::path profileSrcPath;
  boost::filesystem::path debugSrcPath;
  boost::filesystem::path profileDstPath;
  boost::filesystem::path debugDstPath;
};

BOOST_FIXTURE_TEST_SUITE(ProfileCacherTestSuite, ProfileCacherFixture)

BOOST_AUTO_TEST_CASE(testCreationOptionNotSet) {
  // Test that no action is performed if no creation options is set
  auto profileCacher = getProfileCacher();

  profileCacher.storeProfilesToCache();
  profileCacher.restoreProfilesFromCache();
  BOOST_ASSERT(!boost::filesystem::exists(profileDstPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugDstPath));
}

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

BOOST_DATA_TEST_CASE_F(ProfileCacherFixture,
                       testCreationOptionSet,
                       optsCreatingProfile,
                       creationOption) {
  // Test that all the opts creating profiles are working
  auto profileCacher = getProfileCacher("inference", creationOption);

  // Store
  profileCacher.storeProfilesToCache();
  BOOST_ASSERT(boost::filesystem::exists(profileDstPath));
  BOOST_ASSERT(boost::filesystem::exists(debugDstPath));

  // Restore
  // Delete the original files to ensure that we are actually restoring
  boost::filesystem::remove(profileSrcPath);
  boost::filesystem::remove(debugSrcPath);
  BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));
  profileCacher.restoreProfilesFromCache();
  BOOST_ASSERT(boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(boost::filesystem::exists(debugSrcPath));
}

BOOST_AUTO_TEST_CASE(testCustomEngineName) {
  // Test that caching works with custom engineName
  auto profileCacher = getProfileCacher("myEngineName", "autoReport.all");

  // Store
  profileCacher.storeProfilesToCache();
  BOOST_ASSERT(boost::filesystem::exists(profileDstPath));
  BOOST_ASSERT(boost::filesystem::exists(debugDstPath));

  // Restore
  // Delete the original files to ensure that we are actually restoring
  boost::filesystem::remove(profileSrcPath);
  boost::filesystem::remove(debugSrcPath);
  BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));
  profileCacher.restoreProfilesFromCache();
  BOOST_ASSERT(boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(boost::filesystem::exists(debugSrcPath));
}

BOOST_AUTO_TEST_CASE(testCustomAutoReportDir) {
  // Test that caching works with custom autoReportName
  auto profileCacher =
      getProfileCacher("inference", "autoReport.all", "myAutoReportName");

  // Store
  profileCacher.storeProfilesToCache();
  BOOST_ASSERT(boost::filesystem::exists(profileDstPath));
  BOOST_ASSERT(boost::filesystem::exists(debugDstPath));

  // Restore
  // Delete the original files to ensure that we are actually restoring
  boost::filesystem::remove(profileSrcPath);
  boost::filesystem::remove(debugSrcPath);
  BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));
  profileCacher.restoreProfilesFromCache();
  BOOST_ASSERT(boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(boost::filesystem::exists(debugSrcPath));
}

BOOST_AUTO_TEST_CASE(testDebugMissing) {
  // Test that storing and restoring of profiling files works if debug.cbor is
  // missing
  auto profileCacher = getProfileCacher("inference", "autoReport.all");

  // Remove the debug file manually
  boost::filesystem::remove(debugSrcPath);
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));

  // Test storing
  profileCacher.storeProfilesToCache();
  BOOST_ASSERT(boost::filesystem::exists(profileDstPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugDstPath));

  // Delete the original files to ensure that we are actually restoring
  boost::filesystem::remove(profileSrcPath);
  BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
  profileCacher.restoreProfilesFromCache();
  BOOST_ASSERT(boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));
}

BOOST_AUTO_TEST_CASE(testProfileMissing) {
  // Test that storing and restoring is handled correctly if profile.pop is
  // missing
  auto profileCacher = getProfileCacher("inference", "autoReport.all");

  // Remove the debug file manually
  boost::filesystem::remove(profileSrcPath);
  boost::filesystem::remove(debugSrcPath);
  BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));

  // Test storing
  BOOST_CHECK_EXCEPTION(
      profileCacher.storeProfilesToCache(), popart::error, checkErrorMsg);
  BOOST_ASSERT(!boost::filesystem::exists(profileDstPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugDstPath));

  // Delete the original files to ensure that we are not restoring
  BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));
  profileCacher.restoreProfilesFromCache();
  BOOST_ASSERT(!boost::filesystem::exists(profileSrcPath));
  BOOST_ASSERT(!boost::filesystem::exists(debugSrcPath));
}

// End the test suite
BOOST_AUTO_TEST_SUITE_END()
