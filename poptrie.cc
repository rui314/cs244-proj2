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

struct Test {
  uint32_t ip;
  int masklen;
};

extern Test testset[];

using std::chrono::high_resolution_clock;

std::default_random_engine rand_engine;

class Trie;
class Poptrie;
class Poptrie2;

constexpr int K = 6;
constexpr int S = 18;

static constexpr int power_of_two(int n) {
  int x = 1;
  for (int i = 0; i < n; i++)
    x *= 2;
  return x;
}

__attribute__((always_inline))
static inline uint32_t extract(uint32_t bits, int start, int len) {
  return (bits >> (start - len)) & ((1L<<len) - 1);
}

__attribute__((always_inline))
static inline int popcnt(uint64_t x, int len) {
  return __builtin_popcountl(x & ((1UL << len) - 1));
}

class Trie {
public:
  Trie() {
    roots.resize(1<<S);
  }

  void insert(uint32_t key, int key_len, uint32_t val) {
    assert(val < (1 << 30));

    if (key_len <= S) {
      for (int i = 0; i < (1L << (S - key_len)); i++) {
        Node &node = roots[(key >> (32 - S)) + i];
        node.val = val;
        node.is_leaf = true;
      }
      return;
    }

    Node *cur = &roots[extract(key, 32, S)];
    uint32_t bits = extract(key, 32 - S, K);
    int offset = S + K;

    while (offset < key_len) {
      expand(cur);
      cur = &cur->children[bits];
      bits = extract(key, 32 - offset, K);
      offset += K;
    }

    expand(cur);
    for (int i = 0; i < (1L << (offset - key_len)); i++) {
      Node &child = cur->children[bits + i];
      child.val = val;
      child.is_leaf = true;
    }
  }

  uint32_t lookup(uint32_t key) {
    Node *cur = &roots[extract(key, 32, S)];
    int offset = S;
    while (!cur->is_leaf) {
      int bits = extract(key, 32 - offset, K);
      offset += K;
      cur = &cur->children[bits];
    }
    return cur->val;
  }

private:
  friend Poptrie;
  friend Poptrie2;

  struct Node {
    std::vector<Node> children;
    uint32_t val = 0;
    bool is_leaf = true;
  };

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

  uint32_t lookup(uint32_t key) {
    int didx = direct_indices[extract(key, 32, S)];
    if (didx & 0x80000000)
      return didx & 0x7fffffff;

    uint32_t cur = didx;
    uint64_t bits = children[cur].bits;
    uint32_t v = extract(key, 32 - S, K);
    uint32_t offset = S + K;

    while (bits & (1UL << v)) {
      cur = children[cur].base1 + popcnt(bits, v);
      bits = children[cur].bits;
      v = extract(key, 32 - offset, K);
      offset += K;
    } 

    Node c = children[cur];
    int count = __builtin_popcountl(c.leafbits & ((2UL << v) - 1));
    return leaves[c.base0 + count - 1];
 }

  void info() {
    std::cout << "inodes=" << children.size()
              << "\nleaves=" << leaves.size()
              << " size=" << (children.size() * sizeof(children[0]) +
                              leaves.size() * sizeof(leaves[0]) +
                              direct_indices.size() * sizeof(direct_indices[0]))
              << "\n";

    int count[64] = {0};
    for (uint32_t idx : direct_indices)
      if ((idx & 0x80000000) == 0)
        count[__builtin_popcountl(children[idx].bits)]++;
    std::cout << "count:";
    for (int i = 0; i < 64; i++)
      std::cout << " " << count[i];
    std::cout << "\n";

    int count2[64] = {0};
    for (Node &node : children)
      count2[__builtin_popcountl(node.bits)]++;
    std::cout << "count2:";
    for (int i = 0; i < 64; i++)
      std::cout << " " << count2[i];
    std::cout << "\n";
  }

private:
  struct Node {
    uint64_t bits = 0;
    uint64_t leafbits = 0;
    uint32_t base0 = 0;
    uint32_t base1 = 0;
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
  std::vector<uint32_t> leaves;
  std::vector<uint32_t> direct_indices;
};

class Poptrie2 {
  enum {
        DIRECT = 1,
        LEAF_ONLY = 2,
  };

public:
  Poptrie2(Trie &from) {
    direct_indices.resize(1<<S);

    for (int i = 0; i < (1<<S); i++) {
      if (from.roots[i].is_leaf) {
        direct_indices[i] = from.roots[i].val | (DIRECT << 30);
        continue;
      }
      
      if (is_leaf_only(from.roots[i])) {
        direct_indices[i] = leaf_only_node.size() | (LEAF_ONLY << 30);
        import_leaf_only_node(from.roots[i]);
        continue;
      }

      int idx = children.size();
      direct_indices[i] = idx;
      children.push_back({});
      import(from.roots[i], idx);
    }
  }

  uint32_t lookup(uint32_t key) {
    uint32_t didx = direct_indices[extract(key, 32, S)];
    if ((didx >> 30) == DIRECT)
      return didx & 0x3fffffff;

    if ((didx >> 30) == LEAF_ONLY) {
      uint32_t idx = didx & 0x3fffffff;
      uint64_t leafbits = *(uint64_t *)&leaf_only_node[idx];
      uint64_t v = extract(key, 32 - S, K);
      int count = __builtin_popcountl(leafbits & ((2UL << v) - 1));
      return leaf_only_node[idx + count + 1];
    }

    uint32_t cur = didx;
    uint64_t bits = children[cur].bits;
    uint32_t v = extract(key, 32 - S, K);
    uint32_t offset = S + K;

    while (bits & (1UL << v)) {
      cur = children[cur].base1 + popcnt(bits, v);
      bits = children[cur].bits;
      v = extract(key, 32 - offset, K);
      offset += K;
    } 

    Node c = children[cur];
    int count = __builtin_popcountl(c.leafbits & ((2UL << v) - 1));
    return leaves[c.base0 + count - 1];
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
    for (uint32_t idx : direct_indices)
      if ((idx >> 30) == 0)
        count[__builtin_popcountl(children[idx].bits)]++;
    std::cout << "count:";
    for (int i = 0; i < 64; i++)
      std::cout << " " << count[i];
    std::cout << "\n";

    int count2[64] = {0};
    for (Node &node : children)
      count2[__builtin_popcountl(node.bits)]++;
    std::cout << "count2:";
    for (int i = 0; i < 64; i++)
      std::cout << " " << count2[i];
    std::cout << "\n";
  }

private:
  struct Node {
    uint64_t bits = 0;
    uint64_t leafbits = 0;
    uint32_t base0 = 0;
    uint32_t base1 = 0;
  };

  bool is_leaf_only(Trie::Node &node) {
    for (Trie::Node &node : node.children)
      if (!node.is_leaf)
        return false;
    return true;
  }

  void import_leaf_only_node(Trie::Node &node) {
    int start = leaf_only_node.size();
    leaf_only_node.push_back(0);
    leaf_only_node.push_back(0);

    uint64_t leafbits = 1;
    uint32_t last = node.children[0].val;
    leaf_only_node.push_back(last);

    for (size_t i = 1; i < node.children.size(); i++) {
      uint32_t val = node.children[i].val;
      if (val != last) {
        leafbits |= 1L<<i;
        leaf_only_node.push_back(val);
        last = val;
      }
    }

    if (leaf_only_node.size() % 2)
      leaf_only_node.push_back(0);

    *(uint64_t *)&leaf_only_node[start] = leafbits;
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
  std::vector<uint32_t> leaves;
  std::vector<uint32_t> direct_indices;
  std::vector<uint32_t> leaf_only_node;
};

class Mytrie {
public:
  enum {
        len1 = 20,
        len2 = 6,
        len3 = 6,
  };

  Mytrie() {
    nodes.resize(1<<len1);
  }

  void insert(uint32_t key, int key_len, uint32_t val) {
    if (key_len <= len1) { 
      for (int i = 0; i < (1L << (len1 - key_len)); i++) {
        Node &node = nodes[extract(key, 32, len1) + i];
        assert(node.is_direct);
        node.base = val;
      }
      return;
    }

    Node &node = nodes[extract(key, 32, len1)];
    if (node.is_direct)
      expand(node);

    int idx = key & ((1L << (len2 + len3)) - 1);
    for (int i = 0; i < (1L << (32 - key_len)); i++)
      leaves[node.base + idx + i] = val;
  }

  void finalize() {
    std::vector<bool> dup;
    dup.resize(leaves.size() >> len3);

    for (int i = 0; i < (1L << len1); i++) {
      Node &node = nodes[i];
      if (node.is_direct)
        continue;

      int cur = 0;
      int sz = 1L << len3;
      for (int j = 1; j < (1L << len2); j++) {
        if (memcmp(&leaves[node.base + cur * sz], &leaves[node.base + j * sz], sz * sizeof(leaves[0])) == 0) {
          dup[i * (1L << len2) + j] = true;
        } else {
          cur = j;
        }
      }
    }

    std::vector<uint32_t> newidx;
    newidx.resize(leaves.size() >> len3);
    int idx = 0;
    for (size_t i = 0; i < dup.size(); i++) {
      newidx[i] = idx;
      if (!dup[i])
        idx++;
    }
    std::cout << "idx=" << idx << "\n";

    for (int i = 0; i < (1L << len1); i++) {
      Node &node = nodes[i];
      if (node.is_direct)
        continue;

      for (int j = 0; j < (1L << len2); j++)
        if (!dup[i * (1L << len2) + j])
          node.bits = node.bits | (1L << j);
      node.base = newidx[node.base >> len3];
    }

    std::vector<uint32_t> new_leaves;
    for (size_t i = 0; i < dup.size(); i++)
      if (!dup[i])
        for (int j = 0; j < (1L << len3); j++)
          new_leaves.push_back(leaves[i * (1L << len3) + j]);

    std::cout << "leaves.size=" << leaves.size()
              << " new_leaves=" << new_leaves.size()
              << "\n";

    leaves = new_leaves;
  }

  uint32_t lookup(uint32_t key) {
    Node &node = nodes[extract(key, 32, len1)];
    if (node.is_direct)
      return node.base;

    int mid = extract(key, 32 - len1, len2);
    int count = __builtin_popcountl(node.bits & ((2L << mid) - 1));
    int idx1 = (count - 1) * (1L << len3);
    int idx2 = key & ((1L << len3) - 1);
    return leaves[node.base + idx1 + idx2];
 }

  void info() {
    std::cout << "mytrie: node=" << nodes.size()
              << " leaves=" << leaves.size()
              << " size=" << (nodes.size() * sizeof(nodes[0]) + leaves.size() * sizeof(leaves[0]))
              << "\n";
  }

private:
  struct Node {
    uint64_t bits = 0;
    bool is_direct = true;
    uint32_t base = 0;
  };

  void expand(Node &node) {
    assert(node.is_direct);
    node.is_direct = false;

    int val = node.base;
    node.base = leaves.size();
    for (int i = 0; i < (1L << (len2 + len3)); i++)
      leaves.push_back(val);
  }

  std::vector<Node> nodes;
  std::vector<uint32_t> leaves;
};

void assert_(uint32_t expected, uint32_t actual, const std::string &code) {
  if (expected == actual) {
    std::cout << code << " => " << expected << "\n";
  } else {
    std::cout << code << " => " << expected << " expected, but got " << actual << "\n";
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

struct Range {
  uint32_t addr;
  int masklen;
  uint32_t val;
};

static bool in_range(Range &range, uint32_t addr) {
  return range.addr <= addr &&
         addr < range.addr + (1L << (32 - range.masklen));
}

__attribute__((unused))
static Range create_random_range() {
  static std::uniform_int_distribution<uint32_t> dist1(0, UINT32_MAX);
  static std::uniform_int_distribution<uint32_t> dist2(8, 30);
  static std::uniform_int_distribution<uint32_t> dist3(0, 1<<30);

  uint32_t addr = dist1(rand_engine);
  int masklen = dist2(rand_engine);
  uint32_t val = dist3(rand_engine);
  addr = addr & ~((1L << (32 - masklen)) - 1);
  return {addr, masklen, val};
}

__attribute__((unused))
static void test2() {
  std::vector<Range> ranges;
  for (uint32_t i = 0; testset[i].ip && testset[i].masklen; i++)
    ranges.push_back({testset[i].ip, testset[i].masklen, i});

  std::stable_sort(ranges.begin(), ranges.end(),
                   [](const Range &a, const Range &b) {
                     return a.masklen < b.masklen;
                   });

  Trie trie;
  for (Range &range : ranges)
    trie.insert(range.addr, range.masklen, range.val);

  Poptrie ptrie(trie);

  auto find = [&](uint32_t addr) -> uint32_t {
                for (int i = ranges.size() - 1; i >= 0; i--)
                  if (in_range(ranges[i], addr))
                    return ranges[i].val;
                return 0;
              };

  for (Range &range : ranges) {
    uint32_t end = range.addr + (1L << (32 - range.masklen)) - 1;
    // std::cout << "range.addr =" << std::bitset<32>(range.addr) << "/" << range.masklen << "\n";
    // std::cout << "range.addr2=" << std::bitset<32>(end) << "/" << range.masklen << "\n";
    ASSERT(find(range.addr), ptrie.lookup(range.addr));
    ASSERT(find(end), ptrie.lookup(end));
  }
}

__attribute__((unused))
static void test3() {
  std::vector<Range> ranges;
  for (uint32_t i = 0; testset[i].ip && testset[i].masklen; i++)
    ranges.push_back({testset[i].ip, testset[i].masklen, i});

  std::stable_sort(ranges.begin(), ranges.end(),
                   [](const Range &a, const Range &b) {
                     return a.masklen < b.masklen;
                   });

  Mytrie trie;
  for (Range &range : ranges)
    trie.insert(range.addr, range.masklen, range.val);
  trie.finalize();

  auto find = [&](uint32_t addr) -> uint32_t {
                for (int i = ranges.size() - 1; i >= 0; i--)
                  if (in_range(ranges[i], addr))
                    return ranges[i].val;
                return 0;
              };

  for (Range &range : ranges) {
    uint32_t end = range.addr + (1L << (32 - range.masklen)) - 1;
    ASSERT(find(range.addr), trie.lookup(range.addr));
    ASSERT(find(end), trie.lookup(end));
  }
}

class Xorshift {
public:
  Xorshift(uint32_t seed) : state(seed) {}

  uint32_t next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }

private:
  uint32_t state;
};

template <class T>
__attribute__((unused))
static std::chrono::microseconds bench(uint32_t *x, Xorshift &rand, uint64_t repeat) {
  std::vector<Range> ranges;
  for (uint32_t i = 0; testset[i].ip && testset[i].masklen; i++)
    ranges.push_back({testset[i].ip, testset[i].masklen, i});

  std::stable_sort(ranges.begin(), ranges.end(),
                   [](const Range &a, const Range &b) {
                     return a.masklen < b.masklen;
                   });

  Trie trie;
  for (Range &range : ranges)
    trie.insert(range.addr, range.masklen, range.val);

  T ptrie(trie);
  // ptrie.info();

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  uint32_t sum = 0;
  for (uint64_t i = 0; i < repeat; i++)
    sum += ptrie.lookup(rand.next());
  *x = sum;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
}

int main() {
#if 1
  static std::uniform_int_distribution<uint32_t> dist1(0, 1L<<31);
  Xorshift rand(dist1(rand_engine));

  uint32_t sum = 0;
  std::chrono::microseconds dur;
  uint64_t repeat = 10*1000*1000;

  dur = bench<Poptrie>(&sum, rand, repeat);
  dur = bench<Poptrie2>(&sum, rand, repeat);

  dur = bench<Poptrie>(&sum, rand, repeat);
  printf("OK %ld μs\n", dur.count());
  printf("OK %fMlps\n\n", (double)repeat / ((double)dur.count() / 1000 / 1000) / 1000 / 1000);

  dur = bench<Poptrie2>(&sum, rand, repeat);
  printf("OK %ld μs\n", dur.count());
  printf("OK %fMlps\n", (double)repeat / ((double)dur.count() / 1000 / 1000) / 1000 / 1000);

  return sum;
#else
  test2();
  std::cout << "OK\n";
  return 0;
#endif
}
