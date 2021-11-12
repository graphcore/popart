// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REGIONIOMAP_HPP
#define GUARD_NEURALNET_REGIONIOMAP_HPP

#include <memory>
#include <set>
#include <vector>
#include <popart/names.hpp>

// we currently only consider inplacing ops with 1 output. this can be
// generalised in the future if we decide it is necessary

namespace popart {
namespace view {

/// Describes what access an object has to the underlying tensor region.
enum class AccessType { None = 0, Read = 1, Write = 2, ReadWrite = 3, N = 4 };

/**
 * Combine access types.
 *
 * For example,
 * combining AccessType::Read and AccessType::Write would result in
 * AccessType::ReadWrite
 *
 * \param accessTypes Access types to combine.
 * \return The resulting access type.
 **/
AccessType combine(std::set<AccessType> accessTypes);

/**
 * Merge regions and combine the accompanying access types.
 *
 * For example,
 * \c mergeRegions({{{0,0},{3,4}},{{1,3},{3,5}}}) returns
 * \code
 * {{{0,0}, {3,4}},
 *  {{3,3}, {5,4}},
 *  {{1,4}, {5,5}}}
 * \endcode
 *
 * \param regions Region elements to be merged.
 * \return Region elements containing possible region-merges of the
 *         input \p regions.
 **/
Regions mergeRegions(Regions regions);

/**
 * A sub-region of a Shape.
 *
 * The sub-region is an orthotope (or hyperrectangle) of the Shape.
 * A Shape is an object describing the dimensions of a tensor.
 *
 * For example,
 * assume \c A is a tensor with Shape {row, column}.
 * The Region {{0, 0}, {row/2, column}} would then describe the left hand
 * region of \c A, the Region {{row-2, column-3}, {row, column}} would
 * describe a 2x3 sub-tensor in the lower-left corner of \c A and so on.
 * The Region is not restricted to tensors of rank 2, although they
 * are used in this example.
 *
 * \note
 * The upper indices in a region is treated non-inclusive in loops
 **/
class Region {

public:
  /**
   * Region constructor.
   *
   * \param lower_ The indices which mark the start of the region.
   * \param upper_ The indices which mark the end of the region.
   **/
  Region(const std::vector<int64_t> &lower_,
         const std::vector<int64_t> &upper_);

  /**
   * Region constructor.
   *
   * \param lower_ The indices which mark the start of the region.
   * \param upper_ The indices which mark the end of the region.
   * \param accessType The access type that the region has to the underlying
   *                   tensor.
   **/
  Region(const std::vector<int64_t> &lower_,
         const std::vector<int64_t> &upper_,
         const AccessType accessType);

  /// The rank (that is, the number of indices) of the Region.
  int64_t rank() const;

  /// The number of elements in the Region.
  int64_t nelms() const;

  /// Returns true if the number of elements in the region is 0.
  bool isEmpty() const;

  /// Returns the intersecting region with \p rhs.
  Region intersect(const Region &rhs) const;

  /**
   * Return a transposed Region.
   *
   * For example,
   * assume a Region \c R of the form {{l0, l1, l2}, {u0, u1, u2}}.
   * The result of \c R.transpose({1,0,2}) is then {{l1, l0, l2}, {u1, u0, u2}}.
   *
   * \param perm The permutation to apply for the transpose.
   * \return The transposed Region.
   **/
  Region transpose(const Shape perm) const;

  /**
   * Return the reversed Region.
   *
   * For example,
   * assume a Region \c R of the form {{l0, l1, l2}, {u0, u1, u2}}.
   * The result of \c R.reverse({s0,s1,s2},{2,1}) is then
   * {{l0, s1-u1, s2-u2}, {u0, s1-l1, s2-l2}}.
   *
   * \param shape Shape (usually of the tensor) we would like to reverse the
   *              region around.
   * \param dims The dimensions to reverse.
   * \return The reversed region.
   **/
  Region reverse(const Shape shape, const Shape dimensions) const;

  /**
   * Subtract regions from the current region.
   *
   * As the resulting regions must be orthotopes, a cut region which only covers
   * the interior of the current Region will result in Regions with empty
   * elements.
   *
   * The resulting cut includes the last indices of the \p rhs.
   *
   * For example,
   * assume a Region \c R of the form {{0, 0}, {6, 6}}.
   * The result of \c R.sub({{{0,0},{4,4}},{3,3},{7,7}}) is then
   * \code
   * {{{0,4}, {3,6}},
   *  {{4,0}, {6,3}}}
   * \endcode
   *
   * \param rhs Region elements to use in the cut
   * \param include_empty If true, empty regions will be included otherwise they
   *                      will be ignored.
   * \return The regions from the subtraction.
   **/
  Regions sub(const Regions &rhs, bool include_empty = false) const;

  /**
   * Subtract a region from the current region.
   *
   * As the resulting regions must be orthotopes, a cut region which only covers
   * the interior of the current Region will result in Regions with empty
   * elements.
   *
   * The resulting cut includes the last indices of the \p rhs.
   *
   * For example,
   * assume a Region \c R of the form {{0, 0}, {6, 6}}.
   * The result of \c R.sub({0,0},{3,6}) is then {{3, 0}, {6, 6}}
   *
   * \param rhs Region elements to use in the cut.
   * \param include_empty If true, empty regions will be included otherwise they
   *                      will be ignored.
   * \return The regions from the subtraction.
   **/
  Regions sub(const Region &rhs, bool include_empty = false) const;

  /**
   * Return regions resulting from the "cutting" the current region.
   *
   * For example,
   * assume a Region \c R of the form {{2, 1, 0}, {4, 3, 7}}
   * The result of \c R.cut({{3},{2}},false) is then
   * \code
   * {{{3, 1, 0}, {4, 2, 7}},
   *  {{3, 2, 0}, {4, 3, 7}},
   *  {{2, 1, 0}, {3, 2, 7}},
   *  {{2, 2, 0}, {3, 3, 7}}}
   * \endcode
   *
   * \param cuts Where to cut the Region.
   *             The vector index of \c cut correspond to the dimension at where
   *             the Region will be cut.
   *             One cut will result in two output regions.
   *             A subsequent cut will result in four output regions, and so on.
   * \param include_empty If true, empty regions will be included otherwise they
   *                      will be ignored.
   * \return Regions containing the regions after the cuts.
   **/
  Regions cut(const std::vector<std::set<int64_t>> &cuts,
              bool include_empty = false) const;
  /**
   * Return the Region elements resulting from a reshape of the underlying
   * Shape.
   *
   * \param fullInRegion The full region to reshape from.
   * \param fullOutRegion The full region to reshape to.
   * \return All the possible Regions elements which could have resulted from
   *         a reshape of the underlying Shape.
   **/
  Regions reshape(Region fullInRegion, Region fullOutRegion) const;

  /**
   * Return a merged region.
   *
   * \param rhs The Region to merge with.
   * \return A pair containing the merge dimension and the merged Region.
   **/
  std::pair<int64_t, Region> merge(const Region &rhs) const;

  /// Return true if the \p index is fully inside the Region.
  bool contains(const std::vector<int64_t> &index) const;

  /// Return true if \p rhs is fully contained in the Region.
  bool contains(const Region &rhs) const;

  /**
   * Return the flattened index of a non-flattened index relative to the Region.
   *
   * For example,
   * assume a Region \c R of the form {{2, 1, 0}, {4, 3, 7}}, then
   * - \c R.flatIndex({2,1,0}) yields 0 (as this is the start of the region).
   * - \c R.flatIndex({2,1,6}) yields 6 (the last index on the last dimension).
   * - \c R.flatIndex({2,2,0}) yields 7 (first index on the second dimension).
   * - \c R.flatIndex({3,1,0}) yields 14 (first index on the first dimension).
   *
   * \param index The index to get the flattened Region index from.
   * \return The flattened index of the non-flattened index.
   **/
  int64_t flatIndex(const std::vector<int64_t> &index) const;

  /**
   * Return the non-flattened index relative to the Region of a flattened index.
   *
   * For example,
   * assume a Region \c R of the form {{2, 1, 0}, {4, 3, 7}}, then
   * - \c R.dimIndex(0) yields {2,1,0} (as this is the start of the region).
   * - \c R.dimIndex(6) yields {2,1,6} (the last index on the last dimension).
   * - \c R.dimIndex(7) yields {2,2,0} (first index on the second dimension).
   * - \c R.dimIndex(14) yields {3,1,0} (first index on the first dimension).
   *
   * \param index The index to get the full Region indices from.
   * \return The non-flattened index of the flattened index.
   **/
  std::vector<int64_t> dimIndex(int64_t index) const;

  /// Checks that the region is properly constructed and throws an error if not.
  void checks() const;

  /**
   * Return a Region with all elements of \c lower and \c upper equal to 0.
   *
   * \param r The rank of the region.
   *          If set to 0 isEmpty() will return \c true.
   * \return A Region where all of the elements of \c lower and \c upper are
   *         equal to zero.
   **/
  static Region getEmpty(int64_t r);

  /**
   * Get the entire Region spanned by a Shape.
   *
   * For example,
   * the result of \c getFull({1,2,3}) is {{0, 0, 0}, {1, 2, 3}}.
   *
   * \param s The shape to get the full Region from.
   * \param accessType The access type of the region.
   * \return The entire Region.
   **/
  static Region getFull(const Shape &s,
                        AccessType accessType = AccessType::ReadWrite);
  /// Two regions are the same if their \c lower and \c upper indices are the
  /// same.
  bool operator==(const Region &) const;

  /// Two Region objects are different if either their \c lower or \c upper
  /// indices differ.
  bool operator!=(const Region &) const;

  /// Return the \c lower indices.
  const std::vector<int64_t> &getLower() const { return lower; }

  /// Return the \c upper indices.
  const std::vector<int64_t> &getUpper() const { return upper; }

  /// Append the string representation of the current Region to the stream.
  void append(std::ostream &ss) const;

  /// Return the AccessType of the Region.
  AccessType getAccessType() const { return accessType; }

  /// Set the AccessType of the Region.
  void setAccessType(AccessType at) { accessType = at; }

private:
  // The indices which marks the start of the region.
  std::vector<int64_t> lower;
  // The indices which marks the end of the region.
  std::vector<int64_t> upper;
  // rank-0 tensors have no lower and upper bounds,
  // so it is not possible to determine if they are empty
  // by looking for equal lower and upper bounds.
  bool isEmptyRank0{false};

  // Describes what access the Region has to the underlying tensor.
  AccessType accessType{AccessType::None};

  // Private constructor called by the other constructors.
  Region(const std::vector<int64_t> &lower_,
         const std::vector<int64_t> &upper_,
         const AccessType accessType,
         bool isEmpty_r0_);
};

/// Add the string representation of \c r to the stream by calling \c append.
std::ostream &operator<<(std::ostream &stream, const Region &r);

/// Add the string representation of the \c AccessType to the stream.
std::ostream &operator<<(std::ostream &stream, const AccessType &at);

/// Return true if any region is non empty and marked as written to.
bool regionsModified(const view::Regions &regions);

/// Return true if any region is non empty.
bool nonEmptyRegion(const view::Regions &regions);

} // namespace view
} // namespace popart

#endif
