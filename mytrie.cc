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
