// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <boost/filesystem.hpp>
#include <fileoperations.hpp>
#include <sstream>
#include <string>
#include <popart/error.hpp>

namespace popart {

boost::filesystem::path
findFileRecursively(const boost::filesystem::path &searchDir,
                    const std::string &filename) {
  boost::filesystem::path foundPath;
  for (boost::filesystem::directory_entry &curFile :
       boost::filesystem::directory_iterator(searchDir)) {
    if (is_directory(curFile.path())) {
      foundPath = findFileRecursively(curFile.path(), filename);
    } else {
      if (curFile.path().filename() == filename) {
        return curFile.path();
      }
    }
  }
  return foundPath;
}

boost::filesystem::path
rebaseDirHierarchy(const boost::filesystem::path &srcPath,
                   const boost::filesystem::path &srcBaseDir,
                   const boost::filesystem::path &dstBaseDir) {
  std::stringstream ss;
  std::vector<boost::filesystem::path> dstDirStructure;

  auto srcDirFileName = srcBaseDir.filename();
  dstDirStructure.push_back(srcPath.filename());
  auto curDir = srcPath.parent_path();

  // Keep adding directories until it matches the directory of srcBaseDir
  while (curDir.filename() != srcDirFileName) {
    if (curDir.filename() == srcBaseDir.root_path()) {
      ss << "Reached the root directory when searching for common root between "
         << srcPath << " and " << srcBaseDir << "\n";
      throw error(ss.str());
    }
    dstDirStructure.push_back(curDir.filename());
    curDir = curDir.parent_path();
  }

  // Assemble the destination path, starting from dstBaseDir
  auto dstPath = dstBaseDir;
  for (auto it = dstDirStructure.rbegin(); it != dstDirStructure.rend(); ++it) {
    dstPath /= *it;
  }

  return dstPath;
}

} // namespace popart
