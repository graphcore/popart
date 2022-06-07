// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_FILEOPERATIONS_HPP
#define GUARD_FILEOPERATIONS_HPP

#include <boost/filesystem.hpp>
#include <string>

namespace popart {

/**
 * \brief Find a file by recursively searching the search dir.
 *
 * Example:
 *
 * Assume we have the following file structure: ``/foo/bar/foobar/baz.txt``
 * Then ``findFileRecursively(boost::filesystem::path("/foo/bar"), "baz.txt")``
 * would return ``boost::filesystem::path("/foo/bar/foobar/baz.txt")``, whilst
 * ``findFileRecursively(boost::filesystem::path("/foo/bar"),
 * "nonExistent.txt")`` would return ``boost::filesystem::path("")``
 *
 * \param searchDir The directory to search from
 * \param filename The file to search for
 * \return boost::filesystem::path The path to the file searched for.
 *   If none was found, ``boost::filesystem::path("")`` is returned
 */
boost::filesystem::path
findFileRecursively(const boost::filesystem::path &searchDir,
                    const std::string &filename);

/**
 * \brief Change the base part of a path.
 *
 * Useful for getting new destination paths when copying or moving files
 * from a source to a destination.
 *
 * \note No files or directories are modified or created by this function.
 *
 * Example:
 *
 * \code {.cpp}
 * rebaseDirHierarchy(
 *    boost::filesystem::path("/foo/bar/foobar/baz.txt"),
 *    boost::filesystem::path("/foo/bar"),
 *    boost::filesystem::path("/qux"));
 * // Output:
 * // boost::filesystem::path("/qux/foobar/baz.txt")
 * \endcode
 *
 * \param srcPath Path to the source file
 * \param srcBaseDir Path to the directory where the directory hierarchy is
 *   going to be based from
 * \param dstBaseDir Path to the directory where the directory hierarchy and
 *   filename is going to be appended
 \return boost::filesystem::path The resulting destination path
 */
boost::filesystem::path
rebaseDirHierarchy(const boost::filesystem::path &srcPath,
                   const boost::filesystem::path &srcBaseDir,
                   const boost::filesystem::path &dstBaseDir);
} // namespace popart

#endif // GUARD_FILEOPERATIONS_HPP
