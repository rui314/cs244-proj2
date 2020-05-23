#include <iostream>
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

static constexpr int power_of_two(int n) {
  int x = 1;
  for (int i = 0; i < n; i++)
    x *= 2;
  return x;
}

static uint32_t extract(uint32_t bits, int start, int len) {
  return (bits >> (start - len)) & ((1L<<len) - 1);
}

static int popcnt(uint64_t x, int len) {
  return __builtin_popcountl(x & ((1UL << len) - 1));
}

class Trie {
public:
  void insert(uint32_t key, int key_len, uint32_t val) {
    Node *cur = &root;
    uint32_t bits = extract(key, 32, K);
    int offset = K;

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
    Node *cur = &root;
    int offset = 0;
    while (!cur->is_leaf) {
      int bits = extract(key, 32 - offset, K);
      offset += K;
      cur = &cur->children[bits];
    }
    return cur->val;
  }

  void dump() { dump2(root, 0); }

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

  Node root;
};

class Poptrie {
public:
  Poptrie(Trie &trie) {
    children.resize(1);
    import(trie.root, 0);
  }

  uint32_t lookup(uint32_t key) {
    uint32_t cur = 0;
    uint64_t bits = children[0].bits;
    uint32_t offset = 0;
    uint32_t v = extract(key, 32, K);

    while (bits & (1UL << v)) {
      //      std::cout << "=== internal node\n";
      cur = children[cur].base1 + popcnt(bits, v);
      bits = children[cur].bits;
      offset += K;
      v = extract(key, 32 - offset, K);
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
};

void assert_(uint32_t expected, uint32_t actual, const std::string &code) {
  if (expected == actual) {
    std::cout << code << " => " << expected << "\n";
  } else {
    std::cerr << code << " => " << expected << " expected, but got " << actual << "\n";
    exit(1);
  }
}

#define assert(expected, actual) \
  assert_(expected, actual, #actual)

__attribute__((unused))
static void test() {
  Trie trie;
  trie.insert(0, 1, 3);
  trie.insert(0x80000000, 1, 5);
  trie.insert(0x80010000, 16, 8);

  trie.dump();

  assert(3, trie.lookup(0b11));
  assert(3, trie.lookup(0b1));
  assert(3, trie.lookup(0x01234567));
  assert(5, trie.lookup(0x80000010));
  assert(8, trie.lookup(0x80010000));
  assert(8, trie.lookup(0x8001ffff));
  assert(5, trie.lookup(0x80020000));

  Poptrie ptrie(trie);
  ptrie.dump();
  assert(3, ptrie.lookup(0b11));
  assert(3, ptrie.lookup(0b1));
  assert(3, ptrie.lookup(0x01234567));
  assert(5, ptrie.lookup(0x80000010));
  assert(8, ptrie.lookup(0x80010000));
  assert(8, ptrie.lookup(0x8001ffff));
  assert(5, ptrie.lookup(0x80020000));
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

  uint32_t addr = dist1(rand_engine);
  int masklen = dist2(rand_engine);
  uint32_t val = dist1(rand_engine);
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
  // ptrie.dump();

  auto find = [&](uint32_t addr) -> uint32_t {
                for (int i = ranges.size() - 1; i >= 0; i--)
                  if (in_range(ranges[i], addr))
                    return ranges[i].val;
                return 0;
              };

  for (Range &range : ranges) {
    std::cout << "range.addr =" << std::bitset<32>(range.addr) << "/" << range.masklen << "\n";
    std::cout << "range.addr2=" << std::bitset<32>(range.addr + (1L << (32 - range.masklen)) - 1) << "/" << range.masklen << "\n";
    uint32_t expected = find(range.addr);
    assert(expected, ptrie.lookup(range.addr));
    assert(expected, ptrie.lookup(range.addr + (1L << (32 - range.masklen)) - 1));
  }
}

__attribute__((unused))
static std::chrono::microseconds bench(volatile uint32_t *x, std::vector<uint32_t> &random) {
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  std::vector<Range> ranges;
  for (int i = 0; i < 100*1000; i++)
    ranges.push_back(create_random_range());

  std::stable_sort(ranges.begin(), ranges.end(),
                   [](const Range &a, const Range &b) {
                     return a.masklen < b.masklen;
                   });

  Trie trie;
  for (Range &range : ranges)
    trie.insert(range.addr, range.masklen, range.val);
  Poptrie ptrie(trie);


  uint32_t sum = 0;
  for (uint32_t addr : random)
    sum += ptrie.lookup(addr);
  *x = sum;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
}

int main() {
  static std::uniform_int_distribution<uint32_t> dist1(0, UINT32_MAX);
  std::vector<uint32_t> random;
  for (int i = 0; i < 1000*1000; i++)
    random.push_back(dist1(rand_engine));

  volatile uint32_t sum = 0;
  std::chrono::microseconds dur = bench(&sum, random);
  printf("OK %ld μs\n", dur.count());
  return sum;
}
