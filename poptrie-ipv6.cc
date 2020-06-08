#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <random>
#include <bitset>
#include <vector>
#include <cstdlib>
#include <utility>

using std::chrono::high_resolution_clock;

constexpr int K = 6;
constexpr int S = 18;
constexpr int LEN = 128;

typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

struct Range {
  u128 addr;
  int masklen;
  u32 val;
};

extern std::vector<Range> ranges69;
std::default_random_engine rand_engine;

class Trie;
class Poptrie;
class Poptrie2;

static inline u128 extract(u128 bits, int start, int len) {
  return (bits >> (start - len)) & ((1L<<len) - 1);
}

static inline int popcnt(u64 x, int len) {
  return __builtin_popcountl(x & ((1UL << len) - 1));
}

static inline int popcnt_incl(u64 x, int len) {
  return __builtin_popcountl(x & ((2UL << len) - 1));
}

// A normal multi-way trie. It is hard to directly create a Poptrie,
// so we construct a normal multi-way trie first and then convert it
// to a Poptrie.
class Trie {
public:
  struct Node {
    std::vector<Node> children;
    u32 val = 0;
    bool is_leaf = true;
  };

  Trie() {
    roots.resize(1<<S);
  }

  void insert(u128 key, int key_len, u32 val) {
    assert(val < (1 << 30));

    if (key_len <= S) {
      for (int i = 0; i < (1L << (S - key_len)); i++) {
        Node &node = roots[(key >> (LEN - S)) + i];
        node.val = val;
        node.is_leaf = true;
      }
      return;
    }

    Node *cur = &roots[extract(key, LEN, S)];
    u32 bits = extract(key, LEN - S, K);
    int offset = S + K;

    while (offset < key_len) {
      expand(cur);
      cur = &cur->children[bits];
      bits = extract(key, LEN - offset, K);
      offset += K;
    }

    expand(cur);
    for (int i = 0; i < (1L << (offset - key_len)); i++) {
      Node &child = cur->children[bits + i];
      child.val = val;
      child.is_leaf = true;
    }
  }

  u32 lookup(u128 key) {
    Node *cur = &roots[extract(key, LEN, S)];
    int offset = S;
    while (!cur->is_leaf) {
      int bits = extract(key, LEN - offset, K);
      offset += K;
      cur = &cur->children[bits];
    }
    return cur->val;
  }

  void expand(Node *cur) {
    if (!cur->is_leaf)
      return;

    cur->children.resize(1L<<K);
    for (Node &n : cur->children)
      n.val = cur->val;
    cur->is_leaf = false;
  }

  std::vector<Node> roots;
};

// A Poptrie implementation as explained in the paper.
class Poptrie {
public:
  Poptrie(Trie &from) {
    direct_indices.resize(1<<S);

    for (int i = 0; i < (1<<S); i++) {
      if (from.roots[i].is_leaf) {
        direct_indices[i] = from.roots[i].val | 0x80000000;
        continue;
      }

      int idx = children.size();
      direct_indices[i] = idx;
      children.push_back({});
      import(from.roots[i], idx);
    }
  }

  __attribute__((noinline))
  u32 lookup(u128 key) {
    int idx = direct_indices[key >> (LEN - S)];
    if (idx & 0x80000000)
      return idx & 0x7fffffff;

    u32 offset = S;
    u32 v;

    for (;;) {
      Node &node = children[idx];
      v = extract(key, LEN - offset, K);
      if (!(node.bits & (1UL << v)))
        break;
      idx = node.base1 + popcnt(node.bits, v);
      offset += K;
    }

    Node &node = children[idx];
    int count = popcnt_incl(node.leafbits, v);
    return leaves[node.base0 + count - 1];
  }

  void info() {
    std::cout << "inodes=" << children.size()
              << " leaves=" << leaves.size()
              << " size=" << (children.size() * sizeof(children[0]) +
                              leaves.size() * sizeof(leaves[0]) +
                              direct_indices.size() * sizeof(direct_indices[0]))
              << "\n";

    int count[64] = {0};
    for (u32 idx : direct_indices)
      if ((idx & 0x80000000) == 0)
        count[__builtin_popcountl(children[idx].bits)]++;
    std::cout << "dist:";
    for (int i = 0; i < 64; i++)
      std::cout << " " << count[i];
    std::cout << "\n";
  }

private:
  struct Node {
    u64 bits = 0;
    u64 leafbits = 0;
    u32 base0 = 0;
    u32 base1 = 0;
  };

  void import(Trie::Node &from, int idx) {
    assert(from.children.size() == (1<<K));
    int start = children.size();

    for (Trie::Node &node : from.children)
      if (!node.is_leaf)
        children.push_back({});

    Node &to = children[idx];
    to.base0 = leaves.size();
    to.base1 = start;

    for (size_t i = 0; i < from.children.size(); i++)
      if (!from.children[i].is_leaf)
        to.bits |= 1L<<i;

    for (size_t i = 0; i < from.children.size(); i++) {
      if (from.children[i].is_leaf) {
        if (leaves.size() == to.base0 ||
            leaves.back() != from.children[i].val) {
          to.leafbits |= 1L<<i;
          leaves.push_back(from.children[i].val);
        }
      }
    }

    size_t i = 0;
    for (size_t j = 0; j < from.children.size(); j++)
      if (!from.children[j].is_leaf)
        import(from.children[j], start + i++);
  }

  std::vector<Node> children;
  std::vector<u32> leaves;
  std::vector<u32> direct_indices;
};

// A modified version of Poptrie.
class Poptrie2 {
public:
  Poptrie2(Trie &from) {
    direct_indices.resize(1<<S);

    for (int i = 0; i < (1<<S); i++) {
      if (from.roots[i].is_leaf) {
        direct_indices[i] = from.roots[i].val | 0x80000000;
        continue;
      }

      // This is where the modified Poptrie is different from the
      // original one. If a root node has no descendent child internal
      // nodes (i.e. if all children are leaves), the children are
      // stored to a different array.
      if (is_leaf_only(from.roots[i])) {
        direct_indices[i] = leaf_only_node.size() | 0x40000000;
        import_leaf_only_node(from.roots[i]);
        continue;
      }

      int idx = children.size();
      direct_indices[i] = idx;
      children.push_back({});
      import(from.roots[i], idx);
    }
  }

  __attribute__((noinline))
  u32 lookup(u128 key) {
    u32 idx = direct_indices[key >> (LEN - S)];
    if (idx & 0x80000000)
      return idx & 0x3fffffff;

    if (idx & 0x40000000) {
      idx = idx & 0x3fffffff;
      u64 leafbits = *(u64 *)&leaf_only_node[idx];
      u64 v = extract(key, LEN - S, K);
      int count = popcnt_incl(leafbits, v);
      return leaf_only_node[idx + count + 1];
    }

    u32 offset = S;
    u32 v;

    for (;;) {
      Node &node = children[idx];
      v = extract(key, LEN - offset, K);
      if (!(node.bits & (1UL << v)))
        break;
      idx = node.base1 + popcnt(node.bits, v);
      offset += K;
    }

    Node &node = children[idx];
    int count = popcnt_incl(node.leafbits, v);
    return leaves[node.base0 + count - 1];
  }

  void info() {
    std::cout << "inodes=" << children.size()
              << " leaves=" << leaves.size()
              << " size=" << (children.size() * sizeof(children[0]) +
                              leaves.size() * sizeof(leaves[0]) +
                              direct_indices.size() * sizeof(direct_indices[0]) +
                              leaf_only_node.size() * sizeof(leaf_only_node[0]))
              << "\n";

    int count[64] = {0};
    for (u32 idx : direct_indices)
      if ((idx >> 30) == 0)
        count[__builtin_popcountl(children[idx].bits)]++;
    std::cout << "dist:";
    for (int i = 0; i < 64; i++)
      std::cout << " " << count[i];
    std::cout << "\n";
  }

private:
  struct Node {
    u64 bits = 0;
    u64 leafbits = 0;
    u32 base0 = 0;
    u32 base1 = 0;
  };

  bool is_leaf_only(Trie::Node &node) {
    for (Trie::Node &node : node.children)
      if (!node.is_leaf)
        return false;
    return true;
  }

  // Add a leaf-only root node to `leaf_only_node` vector.
  // The bit vector and its leaf values are written to `leaf_only_node`
  // vector next to each other for better locality.
  void import_leaf_only_node(Trie::Node &node) {
    int start = leaf_only_node.size();
    leaf_only_node.push_back(0);
    leaf_only_node.push_back(0);

    u64 leafbits = 1;
    u32 last = node.children[0].val;
    leaf_only_node.push_back(last);

    for (size_t i = 1; i < node.children.size(); i++) {
      u32 val = node.children[i].val;
      if (val != last) {
        leafbits |= 1L<<i;
        leaf_only_node.push_back(val);
        last = val;
      }
    }

    *(u64 *)&leaf_only_node[start] = leafbits;
  }

  void import(Trie::Node &from, int idx) {
    assert(from.children.size() == (1<<K));
    int start = children.size();

    for (Trie::Node &node : from.children)
      if (!node.is_leaf)
        children.push_back({});

    Node &to = children[idx];
    to.base0 = leaves.size();
    to.base1 = start;

    for (size_t i = 0; i < from.children.size(); i++)
      if (!from.children[i].is_leaf)
        to.bits |= 1L<<i;

    for (size_t i = 0; i < from.children.size(); i++) {
      if (from.children[i].is_leaf) {
        if (leaves.size() == to.base0 ||
            leaves.back() != from.children[i].val) {
          to.leafbits |= 1L<<i;
          leaves.push_back(from.children[i].val);
        }
      }
    }

    size_t i = 0;
    for (size_t j = 0; j < from.children.size(); j++)
      if (!from.children[j].is_leaf)
        import(from.children[j], start + i++);
  }

  std::vector<Node> children;
  std::vector<u32> leaves;
  std::vector<u32> direct_indices;
  std::vector<u32> leaf_only_node;
};

class Poptrie3 {
public:
  Poptrie3(Trie &from) {
    direct_indices.resize(1<<S);

    for (int i = 0; i < (1<<S); i++) {
      if (from.roots[i].is_leaf) {
        direct_indices[i] = from.roots[i].val | 0x80000000;
        continue;
      }

      int idx = data.size();
      direct_indices[i] = idx;
      data.resize(data.size() + sizeof(Node));
      import(from.roots[i], idx);
    }
  }

  __attribute__((noinline))
  u32 lookup(u128 key) {
    int idx = direct_indices[key >> (LEN - S)];
    if (idx & 0x80000000)
      return idx & 0x7fffffff;

    u32 offset = S;
    u32 v;

    for (;;) {
      Node &node = *(Node *)&data[idx];
      v = extract(key, LEN - offset, K);
      if (!(node.bits & (1UL << v)))
        break;
      idx = node.base1 + popcnt(node.bits, v) * sizeof(Node);
      offset += K;
    }

    Node &node = *(Node *)&data[idx];
    int count = popcnt_incl(node.leafbits, v);
    return *(u32 *)&data[node.base0 + (count - 1) * 4];
  }

  void info() {}

private:
  struct Node {
    u64 bits = 0;
    u64 leafbits = 0;
    u32 base0 = 0;
    u32 base1 = 0;
  };

  void import(Trie::Node &from, int idx) {
    int nleaves = 0;
    for (Trie::Node &node : from.children)
      if (node.is_leaf)
        nleaves++;

    Node node = {};
    node.base1 = data.size();
    data.resize(data.size() + ((64 - nleaves) * sizeof(Node)));
    node.base0 = data.size();

    u32 last = -1;

    for (size_t i = 0; i < from.children.size(); i++) {
      Trie::Node &child = from.children[i];
      if (!child.is_leaf) {
        node.bits |= 1L<<i;
        continue;
      }

      if (child.val == last)
        continue;

      node.leafbits |= 1L<<i;
      data.resize(data.size() + 4);
      *(u32 *)&data[data.size() - 4] = child.val;
      last = child.val;
    }

    *(Node *)&data[idx] = node;

    size_t i = 0;
    for (size_t j = 0; j < from.children.size(); j++)
      if (!from.children[j].is_leaf)
        import(from.children[j], node.base1 + i++ * sizeof(Node));
  }

  std::vector<uint8_t> data;
  std::vector<u32> direct_indices;
};

class Poptrie4 {
public:
  Poptrie4(Trie &from) {
    direct_indices.resize(1<<S);

    for (int i = 0; i < (1<<S); i++) {
      if (from.roots[i].is_leaf) {
        direct_indices[i] = from.roots[i].val | 0x80000000;
        continue;
      }

      // This is where the modified Poptrie is different from the
      // original one. If a root node has no descendent child internal
      // nodes (i.e. if all children are leaves), the children are
      // stored to a different array.
      if (is_leaf_only(from.roots[i])) {
        direct_indices[i] = data.size() | 0x40000000;
        import_leaf_only_node(from.roots[i]);
        continue;
      }

      int idx = data.size();
      direct_indices[i] = idx;
      data.resize(data.size() + sizeof(Node));
      import(from.roots[i], idx);
    }
  }

  __attribute__((noinline))
  u32 lookup(u128 key) {
    u32 idx = direct_indices[key >> (LEN - S)];
    if (idx & 0x80000000)
      return idx & 0x3fffffff;

    if (idx & 0x40000000) {
      idx = idx & 0x3fffffff;
      u64 leafbits = *(u64 *)&data[idx];
      u64 v = extract(key, LEN - S, K);
      int count = popcnt_incl(leafbits, v);
      return *(u32 *)&data[idx + 8 + (count - 1) * 4];
    }

    u32 offset = S;
    u32 v;

    for (;;) {
      Node &node = *(Node *)&data[idx];
      v = extract(key, LEN - offset, K);
      if (!(node.bits & (1UL << v)))
        break;
      idx = node.base1 + popcnt(node.bits, v) * sizeof(Node);
      offset += K;
    }

    Node &node = *(Node *)&data[idx];
    int count = popcnt_incl(node.leafbits, v);
    return *(u32 *)&data[node.base0 + (count - 1) * 4];
  }

  void info() {
    size_t size = data.size() + direct_indices.size() * sizeof(direct_indices[0]);
    std::cout << " size=" << size << "\n";
  }

private:
  struct Node {
    u64 bits = 0;
    u64 leafbits = 0;
    u32 base0 = 0;
    u32 base1 = 0;
  };

  bool is_leaf_only(Trie::Node &node) {
    for (Trie::Node &node : node.children)
      if (!node.is_leaf)
        return false;
    return true;
  }

  void import_leaf_only_node(Trie::Node &node) {
    int start = data.size();
    data.resize(data.size() + 8);

    u64 leafbits = 1;
    u32 last = node.children[0].val;
    data.resize(data.size() + 4);
    *(u32 *)&data[data.size() - 4] = last;

    for (size_t i = 1; i < node.children.size(); i++) {
      u32 val = node.children[i].val;
      if (val != last) {
        leafbits |= 1L<<i;
        data.resize(data.size() + 4);
        *(u32 *)&data[data.size() - 4] = val;
        last = val;
      }
    }

    *(u64 *)&data[start] = leafbits;
  }

  void import(Trie::Node &from, int idx) {
    int nleaves = 0;
    for (Trie::Node &node : from.children)
      if (node.is_leaf)
        nleaves++;

    Node node = {};
    node.base1 = data.size();
    data.resize(data.size() + ((64 - nleaves) * sizeof(Node)));
    node.base0 = data.size();

    u32 last = -1;

    for (size_t i = 0; i < from.children.size(); i++) {
      Trie::Node &child = from.children[i];
      if (!child.is_leaf) {
        node.bits |= 1L<<i;
        continue;
      }

      if (child.val == last)
        continue;

      node.leafbits |= 1L<<i;
      data.resize(data.size() + 4);
      *(u32 *)&data[data.size() - 4] = child.val;
      last = child.val;
    }

    *(Node *)&data[idx] = node;

    size_t i = 0;
    for (size_t j = 0; j < from.children.size(); j++)
      if (!from.children[j].is_leaf)
        import(from.children[j], node.base1 + i++ * sizeof(Node));
  }

  std::vector<uint8_t> data;
  std::vector<u32> direct_indices;
};

class Poptrie10 {
public:
  Poptrie10(Trie &from) {
    direct_indices.resize(1<<S);

    for (int i = 0; i < (1<<S); i++) {
      if (from.roots[i].is_leaf) {
        direct_indices[i] = from.roots[i].val | 0x80000000;
        continue;
      }

      // This is where the modified Poptrie is different from the
      // original one. If a root node has no descendent child internal
      // nodes (i.e. if all children are leaves), the children are
      // stored to a different array.
      if (is_leaf_only(from.roots[i])) {
        direct_indices[i] = data.size() | 0x40000000;
        import_leaf_only_node(from.roots[i]);
        continue;
      }

      assert(!is_compact(from.roots[i]));

      int idx = data.size();
      direct_indices[i] = idx;
      data.resize(data.size() + sizeof(Node));
      import(from.roots[i], idx);
    }
  }

  __attribute__((noinline))
  u32 lookup(u128 key) {
    u32 idx = direct_indices[key >> (LEN - S)];
    if (idx & 0x80000000)
      return idx & 0x3fffffff;

    if (idx & 0x40000000) {
      idx = idx & 0x3fffffff;
      u64 leafbits = *(u64 *)&data[idx];
      u64 v = extract(key, LEN - S, K);
      int count = popcnt_incl(leafbits, v) * 4;
      return *(u32 *)&data[idx + count + 4];
    }

    u32 offset = S;
    u32 v;

    for (;;) {
      Node &node = *(Node *)&data[idx];
      v = extract(key, LEN - offset, K);
      if (!(node.bits & (1UL << v)))
        break;

      idx = node.base1 + popcnt(node.bits, v) * sizeof(Node);
      offset += K;

      if (node.leafbits & (1UL << v)) {
        u64 leafbits = *(u64 *)&data[idx];
        u64 v = extract(key, LEN - offset, K);
        int count = popcnt_incl(leafbits, v) * 4;
        return *(u32 *)&data[idx + count + 4];
      }
    }

    Node &node = *(Node *)&data[idx];
    u32 bits = ~node.bits & node.leafbits;
    int count = popcnt_incl(bits, v);
    return *(u32 *)&data[node.base0 + (count - 1) * 4];
  }

  void info() {
    std::cout << " size="
              << (data.size() + direct_indices.size() * sizeof(direct_indices[0]))
              << "\n";

    for (int i = 0; i < 63; i++)
      dist[i] += dist[i-1];

    std::cout << " dist=";
    for (int x : dist)
      std::cout << " " << x;
    std::cout << "\n";
  }

private:
  struct Node {
    u64 bits = 0;
    u64 leafbits = 0;
    u32 base0 = 0;
    u32 base1 = 0;
  };

  bool is_leaf_only(Trie::Node &node) {
    for (Trie::Node &node : node.children)
      if (!node.is_leaf)
        return false;
    return true;
  }

  u64 get_leaf_bits(Trie::Node &node) {
    u64 leafbits = 0;
    u32 last = -1;

    for (size_t i = 0; i < 64; i++)  {
      Trie::Node &child = node.children[i];
      if (!child.is_leaf)
        continue;
      if (child.val != last)
        leafbits |= 1L << i;
      last = child.val;
    }
    return leafbits;
  }

  bool is_compact(Trie::Node &node) {
    return !node.is_leaf && is_leaf_only(node) &&
           __builtin_popcountl(get_leaf_bits(node)) <= 4;
  }

  // Add a leaf-only root node to `leaf_only_node` vector.
  // The bit vector and its leaf values are written to `leaf_only_node`
  // vector next to each other for better locality.
  void import_leaf_only_node(Trie::Node &node) {
    int start = data.size();
    data.resize(data.size() + 12);

    u64 leafbits = 1;
    u32 last = node.children[0].val;
    *(u32 *)&data[data.size() - 4] = last;

    for (size_t i = 1; i < node.children.size(); i++) {
      u32 val = node.children[i].val;
      if (val != last) {
        leafbits |= 1L<<i;
        data.resize(data.size() + 4);
        *(u32 *)&data[data.size() - 4] = val;
        last = val;
      }
    }

    *(u64 *)&data[start] = leafbits;
  }

  void import(Trie::Node &from, int idx) {
    if (is_compact(from)) {
      u64 leafbits = get_leaf_bits(from);
      assert(__builtin_popcountl(leafbits) <= 4);
      *(u64 *)&data[idx] = leafbits;
      
      int i = 2;
      for (int j = 0; j < 64; j++)
        if (leafbits & (1L << j))
          *(u32 *)&data[idx + i++ * 4] = from.children[j].val;
      return;
    }

    int nleaves = 0;
    for (Trie::Node &node : from.children)
      if (node.is_leaf)
        nleaves++;

    Node node = {};
    node.leafbits = get_leaf_bits(from);
    node.base1 = data.size();
    data.resize(data.size() + ((64 - nleaves) * sizeof(Node)));
    node.base0 = data.size();

    for (size_t i = 0; i < from.children.size(); i++)
      if (!from.children[i].is_leaf)
        node.bits |= 1L<<i;

    for (int i = 0; i < 64; i++) {
      if (node.leafbits & (1L << i)) {
        data.resize(data.size() + 4);
        *(u32 *)&data[data.size() - 4] = from.children[i].val;
      }
    }

    dist[__builtin_popcountl(node.leafbits)]++;

    for (size_t i = 0; i < from.children.size(); i++)
      if (is_compact(from.children[i]))
        node.leafbits |= 1L<<i;

    *(Node *)&data[idx] = node;

    size_t i = 0;
    for (size_t j = 0; j < from.children.size(); j++)
      if (!from.children[j].is_leaf)
        import(from.children[j], node.base1 + i++ * sizeof(Node));
  }

  int dist[64] = {0};

  std::vector<u32> direct_indices;
  std::vector<uint8_t> data;
};

void assert_(u128 expected, u128 actual, const std::string &code) {
  if (expected == actual) {
    std::cout << code << " => " << (uint64_t)(expected>>64) << (uint64_t)(expected)
              << "\n";
  } else {
    std::cout << code << " => " << (uint64_t)(expected>>64) << (uint64_t)(expected)
              << " expected, but got "
              << (uint64_t)(actual>>64) << (uint64_t)(actual) << "\n";
    exit(1);
  }
}

#define ASSERT(expected, actual) \
  assert_(expected, actual, #actual)

__attribute__((unused))
static void test1() {
  Trie trie;
  trie.insert(0, 1, 3);
  trie.insert(0x80000000, 1, 5);
  trie.insert(0x80010000, 16, 8);

  ASSERT(3, trie.lookup(0b11));
  ASSERT(3, trie.lookup(0b1));
  ASSERT(3, trie.lookup(0x01234567));
  ASSERT(5, trie.lookup(0x80000010));
  ASSERT(8, trie.lookup(0x80010000));
  ASSERT(8, trie.lookup(0x8001ffff));
  ASSERT(5, trie.lookup(0x80020000));

  Poptrie ptrie(trie);
  ASSERT(3, ptrie.lookup(0b11));
  ASSERT(3, ptrie.lookup(0b1));
  ASSERT(3, ptrie.lookup(0x01234567));
  ASSERT(5, ptrie.lookup(0x80000010));
  ASSERT(8, ptrie.lookup(0x80010000));
  ASSERT(8, ptrie.lookup(0x8001ffff));
  ASSERT(5, ptrie.lookup(0x80020000));
}

static bool in_range(Range &range, u128 addr) {
  return range.addr <= addr &&
         addr < range.addr + (1L << (LEN - range.masklen));
}

__attribute__((unused))
static void test() {
  Trie trie;
  for (Range &range : ranges69)
    trie.insert(range.addr, range.masklen, range.val);

  Poptrie10 ptrie(trie);

  auto find = [&](u128 addr) -> u128 {
                for (int i = ranges69.size() - 1; i >= 0; i--)
                  if (in_range(ranges69[i], addr))
                    return ranges69[i].val;
                return 0;
              };

  for (Range &range : ranges69) {
    u128 end = range.addr + (1L << (LEN - range.masklen)) - 1;
    ASSERT(find(range.addr), ptrie.lookup(range.addr));
    ASSERT(find(end), ptrie.lookup(end));
  }
}

class Xorshift {
public:
  Xorshift(u32 a, u32 b, u32 c, u32 d) : a(a), b(b), c(c), d(d) {}

  u128 next() {
    u32 t = d;
    u32 s = a;
    d = c;
    c = b;
    b = s;

    t ^= t << 11;
    t ^= t >> 8;
    a = t ^ s ^ (s >> 19);
    return ((u128)a<<96) | ((u128)b<<64) | ((u128)c<<32) | d;
  }

private:
  u32 a, b, c, d;
};

template <class T>
__attribute__((unused))
static std::chrono::microseconds bench(Xorshift rand, u64 repeat, bool show_info) {
  Trie trie;
  for (Range &range : ranges69)
    trie.insert(range.addr, range.masklen, range.val);

  T ptrie(trie);
  if (show_info)
    ptrie.info();

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (u64 i = 0; i < repeat; i++) {
    u128 addr = ((u128)0x20 << 120) | (rand.next() >> 120);
    ptrie.lookup(addr);
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
}

int main() {
  std::stable_sort(ranges69.begin(), ranges69.end(),
                   [](const Range &a, const Range &b) {
                     return a.masklen < b.masklen;
                   });

#if 1
  static std::uniform_int_distribution<u32> dist1(0, 1L<<31);
  Xorshift rand(dist1(rand_engine), dist1(rand_engine),
                dist1(rand_engine), dist1(rand_engine));

  std::chrono::microseconds dur;
  u64 repeat = 300*1000*1000;

  std::cout << "Look up random " << repeat << " keys for each test. "
            << "S=" << S << " K=" << K << "\n";

  //  dur = bench<Poptrie>(rand, repeat, false);
  //  dur = bench<Poptrie2>(rand, repeat, false);

  std::cout << " Original: ";
  dur = bench<Poptrie>(rand, repeat, false);
  printf("%.1f Mlps\n", (double)repeat / (double)dur.count());

  std::cout << "Leaf-only: ";
  dur = bench<Poptrie2>(rand, repeat, false);
  printf("%.1f Mlps\n", (double)repeat / (double)dur.count());

  std::cout << "   Layout: ";
  dur = bench<Poptrie3>(rand, repeat, false);
  printf("%.1f Mlps\n", (double)repeat / (double)dur.count());

  std::cout << "     Both: ";
  dur = bench<Poptrie4>(rand, repeat, false);
  printf("%.1f Mlps\n", (double)repeat / (double)dur.count());

  return 0;
#else
  test();
  std::cout << "OK\n";
  return 0;
#endif
}
