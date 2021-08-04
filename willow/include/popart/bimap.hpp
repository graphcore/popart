// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BIMAP_HPP
#define GUARD_NEURALNET_BIMAP_HPP

#include <map>

template <typename L, typename R> class BiMap {

public:
  void insert(L l, R r) {
    left[l]  = r;
    right[r] = l;
  }

  bool hasLeft(const L &l) const { return left.find(l) != left.end(); }

  bool hasRight(const R &r) const { return right.find(r) != right.end(); }

  void eraseLeft(L l) {
    R &r = left.at(l);
    left.erase(l);
    right.erase(r);
  }

  void eraseRight(R r) {
    L &l = right.at(r);
    left.erase(l);
    right.erase(r);
  }

  void remapLeft(L from, L to) {
    R r = left.at(from);
    eraseLeft(from);
    insert(to, r);
  }

  void remapRight(R from, R to) {
    L l = right.at(from);
    eraseRight(from);
    insert(l, to);
  }

  const R &operator[](L l) const { return left.at(l); }

  const R &at(L l) const { return left.at(l); }

  const L &getLeft(R r) const { return right.at(r); }

  const R &getRight(L l) const { return left.at(l); }

  const std::map<L, R> &leftMap() const { return left; }

  const std::map<R, L> &rightMap() const { return right; }

private:
  std::map<L, R> left;
  std::map<R, L> right;
};

#endif
