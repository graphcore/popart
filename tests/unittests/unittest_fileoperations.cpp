// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE FileOperations

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/test/unit_test.hpp>
#include <fileoperations.hpp>
#include <fstream>
#include <popart/error.hpp>

struct FileOperationsFixture {
  FileOperationsFixture()
      : tmpDirRoot(boost::filesystem::temp_directory_path() /
                   boost::filesystem::unique_path()) {
    // Create tmp directory
    boost::filesystem::create_directories(tmpDirRoot);
  }
  ~FileOperationsFixture() {
    // Clean-up
    boost::filesystem::remove_all(tmpDirRoot);
  }

  boost::filesystem::path tmpDirRoot;
};

struct FileOperationsErrorFixture {
  bool checkErrorMsg(const popart::error &ex) {
    const auto expectedPrefix = "Reached the root directory when";
    return boost::algorithm::starts_with(ex.what(), expectedPrefix);
  }
};

BOOST_FIXTURE_TEST_SUITE(FileOperationsTestSuite, FileOperationsFixture)

BOOST_AUTO_TEST_CASE(testFindFileRecursively) {
  // Test that testFindFileRecursively is able to find files

  // Create the directory hierarchy
  auto tmpDirLeaf = tmpDirRoot / boost::filesystem::unique_path() /
                    boost::filesystem::unique_path();
  std::string tmpFileName = "tmpFile.txt";
  auto tmpFile            = tmpDirLeaf / tmpFileName;
  // Create an empty file
  boost::filesystem::create_directories(tmpFile.parent_path());
  std::ofstream output(tmpFile.string());

  // Test functionality
  auto foundFile = popart::findFileRecursively(tmpDirRoot, tmpFileName);
  BOOST_ASSERT(foundFile == tmpFile);
}

BOOST_AUTO_TEST_CASE(testFindFileRecursivelyNeg) {
  // Test that testFindFileRecursively returns an empty path if not found
  auto notFoundFile =
      popart::findFileRecursively(tmpDirRoot, "nonExistentFile.txt");
  BOOST_ASSERT(notFoundFile == boost::filesystem::path());
}

// End the test suite
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_CASE(testRebaseDirHierarchy) {
  // Test that appropriate destination paths can be created
  auto dstPath = popart::rebaseDirHierarchy(
      boost::filesystem::path("/foo/bar/foobar/baz.txt"),
      boost::filesystem::path("/foo/bar"),
      boost::filesystem::path("/qux"));

  BOOST_ASSERT(dstPath == boost::filesystem::path("/qux/foobar/baz.txt"));
}

BOOST_FIXTURE_TEST_SUITE(FileOperationsErrorTestSuite,
                         FileOperationsErrorFixture)

BOOST_AUTO_TEST_CASE(testRebaseDirHierarchyNeg) {
  // Test that an error is raised if srcPath and srcBaseDir does not share
  // common root

  BOOST_CHECK_EXCEPTION(popart::rebaseDirHierarchy(
                            boost::filesystem::path("/foo/bar/foobar/baz.txt"),
                            boost::filesystem::path("/quux"),
                            boost::filesystem::path("/qux")),
                        popart::error,
                        checkErrorMsg);
}

// End the test suite
BOOST_AUTO_TEST_SUITE_END()
