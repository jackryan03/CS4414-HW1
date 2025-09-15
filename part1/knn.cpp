#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}

// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T,int>>& items, int depth) {
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    if (items.empty()) return nullptr;

    const size_t dim = 1;
    const int axis = depth % dim;

    auto cmp = [axis, dim](const std::pair<Embedding_T,int> &a, const std::pair<Embedding_T,int> &b) {
        for (size_t offset = 0; offset < dim; ++offset) {
            size_t cur = (axis + offset) % dim;

            float n1 = getCoordinate(a.first, cur);
            float n2 = getCoordinate(b.first, cur);
            
            if (n1 < n2) return true;
            if (n1 > n2) return false;
        }
        return false;
    };

    std::sort(items.begin(), items.end(), cmp);

    int mid = items.size() / 2;

    Node *node = new Node(items[mid].first, items[mid].second);

    std::vector<std::pair<Embedding_T,int>> leftItems(items.begin(), items.begin() + mid);
    std::vector<std::pair<Embedding_T,int>> rightItems(items.begin() + mid + 1, items.end());

    node->left = buildKD(leftItems, depth + 1);
    node->right = buildKD(rightItems, depth + 1);

    return node;
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */
    const size_t dim = 1;
    int axis = depth % dim;

    float queryCoord = getCoordinate(Node::queryEmbedding, axis);
    float nodeCoord = getCoordinate(node->embedding, axis);

    float dist = distance(Node::queryEmbedding, node->embedding);

    if (heap.size() < static_cast<size_t>(K)) {
        heap.emplace(dist, node->idx);
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.emplace(dist, node->idx);
    }

    Node *nextNode = (queryCoord < nodeCoord) ? node->left : node->right;
    Node *otherNode = (queryCoord < nodeCoord) ? node->right : node->left;

    if (nextNode) {
        knnSearch(nextNode, depth + 1, K, heap);
    }

    if (otherNode) {
        if (heap.size() < static_cast<size_t>(K) || std::abs(queryCoord - nodeCoord) < heap.top().first) {
            knnSearch(otherNode, depth + 1, K, heap);
        }
    }

    return;
}