/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_PUCK_INDEX_SERVICE_H
#define OPENSEARCH_KNN_PUCK_INDEX_SERVICE_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

// Forward declare Puck index types
namespace puck {
    class HierarchicalClusterIndex;
}

namespace knn_jni {
namespace puck_index_service {

/**
 * Puck index service for managing hierarchical cluster indices
 */
class PuckIndexService {
public:
    /**
     * Create a new Puck hierarchical cluster index
     *
     * @param numDocs Expected number of documents
     * @param dimension Vector dimension
     * @param parameters Index parameters
     * @return Unique pointer to the created index
     */
    static std::unique_ptr<puck::HierarchicalClusterIndex> createIndex(
        long numDocs,
        int dimension,
        const std::unordered_map<std::string, std::string>& parameters
    );

    /**
     * Train the index with hierarchical clustering
     *
     * @param index Pointer to the index
     * @param vectors Training vectors
     * @param numVectors Number of training vectors
     * @param dimension Vector dimension
     * @param parameters Training parameters
     */
    static void trainIndex(
        puck::HierarchicalClusterIndex* index,
        const std::vector<std::vector<float>>& vectors,
        int numVectors,
        int dimension,
        const std::unordered_map<std::string, std::string>& parameters
    );

    /**
     * Query the index for nearest neighbors
     *
     * @param index Pointer to the index
     * @param queryVector Query vector
     * @param k Number of nearest neighbors
     * @param parameters Query parameters
     * @return Vector of (distance, index) pairs
     */
    static std::vector<std::pair<float, long>> queryIndex(
        puck::HierarchicalClusterIndex* index,
        const std::vector<float>& queryVector,
        int k,
        const std::unordered_map<std::string, std::string>& parameters
    );
};

} // namespace puck_index_service
} // namespace knn_jni

#endif // OPENSEARCH_KNN_PUCK_INDEX_SERVICE_H