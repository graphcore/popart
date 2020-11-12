// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// Suffix Tree implemenation based on:
// http://llvm.org/doxygen/MachineOutliner_8cpp_source.html
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

#include <popart/subgraph/suffixtree.hpp>

namespace fwtools {
namespace subgraph {
namespace suffixtree {

namespace {

const int emptyIdx = std::numeric_limits<int>::max();
struct Node {
  std::map<int, Node *> children;
  int startIdx = emptyIdx;
  int *endIdx  = nullptr;
  Node *link   = nullptr;
  bool isRoot() const { return startIdx == emptyIdx; }
  size_t size() const {
    if (isRoot()) {
      return 0;
    }
    return *endIdx - startIdx + 1;
  }

  Node(int sIn, int *eIn, Node *lIn) : startIdx(sIn), endIdx(eIn), link(lIn) {}

  int suffixIdx = emptyIdx;
  int concatLen = 0;
  // int nDescendents = 0;
  std::vector<int> descendantStartIndices = {};
  int nChildrenWaitingFor                 = 0;
  Node *parent                            = nullptr;
};

class SuffixTree {
public:
  const std::vector<int> &sequence;
  std::map<int, std::unique_ptr<Node>> nodeMap;
  int nodeMapSize = 0;
  Node *root      = nullptr;

private:
  std::map<int, int> internalEndIdxAllocator;
  int internalEndIdxAllocatorSize = 0;

  int leafEndIdx = std::numeric_limits<int>::max();

  struct ActiveState {
    Node *node;
    int Idx = emptyIdx;
    int Len = 0;
  };
  ActiveState Active;

  Node *insertLeaf(Node &parent, int startIdx, int edge) {
    nodeMap[nodeMapSize] =
        std::unique_ptr<Node>(new Node(startIdx, &leafEndIdx, nullptr));
    Node *N = nodeMap[nodeMapSize].get();
    nodeMapSize++;
    parent.children[edge] = N;
    return N;
  }
  Node *insertInternalNode(Node *parent, int startIdx, int endIdx, int edge) {

    internalEndIdxAllocator[internalEndIdxAllocatorSize] = endIdx;
    int *E = &internalEndIdxAllocator[internalEndIdxAllocatorSize];
    internalEndIdxAllocatorSize++;

    nodeMap[nodeMapSize] = std::unique_ptr<Node>(new Node(startIdx, E, root));

    Node *N = nodeMap[nodeMapSize].get();
    nodeMapSize++;

    if (parent) {
      parent->children[edge] = N;
    }

    return N;
  }

  void setSuffixIndices(Node &currNode, int currNodeLen) {

    bool isLeaf        = currNode.children.size() == 0 && !currNode.isRoot();
    currNode.concatLen = currNodeLen;
    for (auto &childPair : currNode.children) {
      setSuffixIndices(*childPair.second,
                       currNodeLen +
                           static_cast<int>(childPair.second->size()));
    }

    if (isLeaf) {
      currNode.suffixIdx = static_cast<int>(sequence.size()) - currNodeLen;
    }
  }

  void setNDescendents() {
    std::vector<Node *> ready;
    for (auto &nodePair : nodeMap) {

      Node *n                = nodePair.second.get();
      n->nChildrenWaitingFor = static_cast<int>(n->children.size());
      for (auto &childPair : n->children) {
        Node *child   = childPair.second;
        child->parent = n;
      }
      if (n->nChildrenWaitingFor == 0) {
        n->descendantStartIndices = {n->suffixIdx};
        ready.push_back(n);
      }
    }

    while (ready.size() != 0) {
      Node *n = ready.back();
      ready.resize(ready.size() - 1);
      if (n->parent) {
        n->parent->nChildrenWaitingFor--;
        for (auto &l : n->descendantStartIndices) {
          n->parent->descendantStartIndices.push_back(l);
        }
        if (n->parent->nChildrenWaitingFor == 0) {
          ready.push_back(n->parent);
        }
      }
    }
  }

  int extend(int endIdx, int nSuffixesToAdd) {
    Node *Needslink = nullptr;
    while (nSuffixesToAdd > 0) {
      if (Active.Len == 0) {
        Active.Idx = endIdx;
      }
      int FirstChar = sequence[Active.Idx];
      if (Active.node->children.count(FirstChar) == 0) {
        insertLeaf(*Active.node, endIdx, FirstChar);
        if (Needslink) {
          Needslink->link = Active.node;
          Needslink       = nullptr;
        }
      } else {
        Node *NextNode   = Active.node->children[FirstChar];
        int SubstringLen = static_cast<int>(NextNode->size());
        if (Active.Len >= SubstringLen) {
          Active.Idx += SubstringLen;
          Active.Len -= SubstringLen;
          Active.node = NextNode;
          continue;
        }
        int LastChar = sequence[endIdx];
        if (sequence[NextNode->startIdx + Active.Len] == LastChar) {
          if (Needslink && !Active.node->isRoot()) {
            Needslink->link = Active.node;
            Needslink       = nullptr;
          }

          Active.Len++;
          break;
        }
        Node *SplitNode =
            insertInternalNode(Active.node,
                               NextNode->startIdx,
                               NextNode->startIdx + Active.Len - 1,
                               FirstChar);
        insertLeaf(*SplitNode, endIdx, LastChar);
        NextNode->startIdx += Active.Len;
        SplitNode->children[sequence[NextNode->startIdx]] = NextNode;
        if (Needslink)
          Needslink->link = SplitNode;

        Needslink = SplitNode;
      }
      nSuffixesToAdd--;
      if (Active.node->isRoot()) {
        if (Active.Len > 0) {
          Active.Len--;
          Active.Idx = endIdx - nSuffixesToAdd + 1;
        }
      } else {
        Active.node = Active.node->link;
      }
    }

    return nSuffixesToAdd;
  }

public:
  SuffixTree(const std::vector<int> &seq0) : sequence(seq0) {
    root               = insertInternalNode(nullptr, emptyIdx, emptyIdx, 0);
    Active.node        = root;
    int nSuffixesToAdd = 0;
    for (int PfxendIdx = 0, End = static_cast<int>(sequence.size());
         PfxendIdx < End;
         PfxendIdx++) {
      nSuffixesToAdd++;
      leafEndIdx     = PfxendIdx;
      nSuffixesToAdd = extend(PfxendIdx, nSuffixesToAdd);
    }
    setSuffixIndices(*root, 0);
    setNDescendents();
  }
};

} // namespace

std::vector<Match> getInternal(const std::vector<int> &s) {
  SuffixTree tree(s);

  // matches[i] will be be all internal nodes with concatLen = i
  std::vector<std::vector<Match>> matches(s.size() + 1);

  for (auto &x : tree.nodeMap) {
    auto node = x.second.get();
    // not a leaf, not the root
    if (node->children.size() != 0 && node->parent != nullptr) {
      matches[node->concatLen].push_back(
          {node->descendantStartIndices, node->concatLen});
    }
  }

  std::vector<Match> final_matches;
  final_matches.reserve(s.size());
  for (int i = static_cast<int>(s.size()); i >= 1; --i) {
    if (matches[i].size() > 0) {
      final_matches.insert(
          final_matches.end(), matches[i].begin(), matches[i].end());
    }
  }
  return final_matches;
}

} // namespace suffixtree
} // namespace subgraph
} // namespace fwtools
