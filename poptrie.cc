#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <utility>

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

  int lookup(uint32_t key) {
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
  struct Node {
    std::vector<Node> children;
    uint32_t val = 0;
    bool is_leaf = true;
  };

  void expand(Node *cur) {
    if (!cur->is_leaf)
      return;

    cur->children.resize(1<<K);
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
  Poptrie() {
    children.push_back({0, 0, 0, 0});
    leaves.push_back(0);
  }

  bool insert(uint32_t key, uint32_t key_len, uint32_t val);

  uint32_t lookup(uint32_t key) {
    uint32_t cur = 0;
    uint64_t bits = children[0].bits;
    uint32_t offset = 0;
    uint32_t v = extract(key, 0, K);

    while (bits & (1UL << v)) {
      cur = children[cur].base1 + popcnt(bits, v);
      bits = children[cur].bits;
      offset += K;
      v = extract(key, offset, K);
    }

    Node c = children[cur];
    return leaves[c.base0 + popcnt(c.leafbits, v + 1)];
 }

private:
  struct Node {
    uint64_t bits;
    uint64_t leafbits;
    uint32_t base0;
    uint32_t base1;
  };

  std::vector<Node> children;
  std::vector<uint32_t> leaves;
};

void assert_(int expected, int actual, const std::string &code) {
  if (expected == actual) {
    std::cout << code << " => " << expected << "\n";
  } else {
    std::cerr << code << " => " << expected << " expected, but got " << actual << "\n";
    exit(1);
  }
}

#define assert(expected, actual) \
  assert_(expected, actual, #actual)

static void test() {
  Trie trie;
  trie.insert(0, 1, 3);
  trie.insert(0x80000000, 1, 5);
  trie.insert(0x80010000, 16, 8);

  // trie.dump();

  assert(3, trie.lookup(0b11));
  assert(3, trie.lookup(0b1));
  assert(3, trie.lookup(0x01234567));
  assert(5, trie.lookup(0x80000010));
  assert(8, trie.lookup(0x80010000));
  assert(8, trie.lookup(0x8001ffff));
  assert(5, trie.lookup(0x80020000));
}

int main() {
  test();
  std::cout << "OK\n";
}
