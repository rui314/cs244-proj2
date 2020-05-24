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

std::default_random_engine rand_engine;

class Trie;
class Poptrie;

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

  void dump() {
    for (Node &node : roots)
      dump2(node, 0);
  }

private:
  friend Poptrie;

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

  void dump2(Node &cur, int indent) {
    if (cur.is_leaf) {
      std::cout << std::string(indent, ' ') << cur.val << "\n";
      return;
    }
    for (Node &n : cur.children)
      dump2(n, indent + 2);
  };

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
    /*
    std::cout << "=== leaf: v=" << v
              << " c.base0=" << c.base0
              << " popcnt=" << count
              << " leaves[c.base0 + popcnt(c.leafbits, v + 1) - 1]=" << leaves[c.base0 + count - 1]
              << "\n";
    */
    return leaves[c.base0 + count - 1];
 }

  void info() {
    std::cout << "inodes=" << children.size()
              << "\nleaves=" << leaves.size()
              << "\n";
  }

  void dump() {
    std::cout << "Children:\n";
    for (Node &node : children)
      std::cout << " bits=" << std::bitset<64>(node.bits)
                << " leafbits=" << std::bitset<64>(node.leafbits)
                << " base0=" << node.base0 << " base1=" << node.base1 << "\n";
    std::cout << "Leaves:";
    for (uint32_t x : leaves)
      std::cout << " " << x;
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
static void test() {
  Trie trie;
  trie.insert(0, 1, 3);
  trie.insert(0x80000000, 1, 5);
  trie.insert(0x80010000, 16, 8);

  // trie.dump();

  ASSERT(3, trie.lookup(0b11));
  ASSERT(3, trie.lookup(0b1));
  ASSERT(3, trie.lookup(0x01234567));
  ASSERT(5, trie.lookup(0x80000010));
  ASSERT(8, trie.lookup(0x80010000));
  ASSERT(8, trie.lookup(0x8001ffff));
  ASSERT(5, trie.lookup(0x80020000));

  Poptrie ptrie(trie);
  ptrie.dump();
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
  for (int i = 0; i < 110; i++)
    ranges.push_back(create_random_range());

  std::stable_sort(ranges.begin(), ranges.end(),
                   [](const Range &a, const Range &b) {
                     return a.masklen < b.masklen;
                   });

  Trie trie;
  for (Range &range : ranges)
    trie.insert(range.addr, range.masklen, range.val);

  Poptrie ptrie(trie);
  ptrie.dump();

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

static constexpr int repeat = 10;

__attribute__((unused))
static std::chrono::microseconds bench(uint32_t *x, std::vector<uint32_t> &random) {
  std::vector<Range> ranges;
  for (int i = 0; i < 84000; i++)
    ranges.push_back(create_random_range());

  std::stable_sort(ranges.begin(), ranges.end(),
                   [](const Range &a, const Range &b) {
                     return a.masklen < b.masklen;
                   });

  Trie trie;
  for (Range &range : ranges)
    trie.insert(range.addr, range.masklen, range.val);

  Poptrie ptrie(trie);
  ptrie.info();

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  uint32_t sum = 0;
  for (int i = 0; i < repeat; i++)
    for (uint32_t addr : random)
      sum += ptrie.lookup(addr);
  *x = sum;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
}

int main() {
#if 1
  static std::uniform_int_distribution<uint32_t> dist1(0, 1<<30);
  std::vector<uint32_t> random;
  for (int i = 0; i < 10*1000*1000; i++)
    random.push_back(dist1(rand_engine));

  uint32_t sum = 0;
  std::chrono::microseconds dur = bench(&sum, random);
  printf("OK %ld μs\n", dur.count());
  printf("OK %fMlps\n", (double)(random.size() * repeat) / ((double)dur.count() / 1000 / 1000) / 1000 / 1000);
  return sum;
#else
  test2();
  std::cout << "OK\n";
  return 0;
#endif
}
