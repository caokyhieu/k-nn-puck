/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "puck_index_service.h"

// Include Puck headers
#include "puck/hierarchical_cluster/hierarchical_cluster_index.h"
#include "puck/index.h"
#include "puck/index_conf.h"

#include <memory>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace knn_jni {
namespace puck_index_service {

std::unique_ptr<puck::HierarchicalClusterIndex> PuckIndexService::createIndex(
    long numDocs,
    int dimension,
    const std::unordered_map<std::string, std::string>& parameters) {

    try {
        // Create Puck hierarchical cluster index
        auto index = std::make_unique<puck::HierarchicalClusterIndex>();

        if (!index) {
            throw std::runtime_error("Failed to create HierarchicalClusterIndex");
        }

        // Initialize the index
        int result = index->init();
        if (result != 0) {
            throw std::runtime_error("Failed to initialize Puck index, error code: " + std::to_string(result));
        }

        return index;

    } catch (const std::exception& e) {
        throw std::runtime_error("Error creating Puck index: " + std::string(e.what()));
    }
}

void PuckIndexService::trainIndex(
    puck::HierarchicalClusterIndex* index,
    const std::vector<std::vector<float>>& vectors,
    int numVectors,
    int dimension,
    const std::unordered_map<std::string, std::string>& parameters) {

    try {
        if (!index) {
            throw std::invalid_argument("Index pointer is null");
        }

        if (vectors.empty()) {
            throw std::invalid_argument("Training vectors are empty");
        }

        // Puck expects training to be performed through the train() method
        // The actual training data would typically be loaded from configuration
        // For now, we'll call the train method with the expectation that
        // training data is configured elsewhere

        int result = index->train();
        if (result != 0) {
            throw std::runtime_error("Failed to train Puck index, error code: " + std::to_string(result));
        }

        // After training, build the index
        result = index->build();
        if (result != 0) {
            throw std::runtime_error("Failed to build Puck index, error code: " + std::to_string(result));
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Error training Puck index: " + std::string(e.what()));
    }
}

std::vector<std::pair<float, long>> PuckIndexService::queryIndex(
    puck::HierarchicalClusterIndex* index,
    const std::vector<float>& queryVector,
    int k,
    const std::unordered_map<std::string, std::string>& parameters) {

    try {
        if (!index) {
            throw std::invalid_argument("Index pointer is null");
        }

        if (queryVector.empty()) {
            throw std::invalid_argument("Query vector is empty");
        }

        // Create Puck request and response objects
        puck::Request request;
        puck::Response response;

        // Set up request
        request.topk = k;
        request.feature = queryVector.data();

        // Allocate response arrays
        std::vector<float> distances(k);
        std::vector<uint32_t> indices(k);

        response.distance = distances.data();
        response.local_idx = indices.data();
        response.result_num = 0;

        // Perform the search
        int result = index->search(&request, &response);
        if (result != 0) {
            throw std::runtime_error("Failed to search Puck index, error code: " + std::to_string(result));
        }

        // Convert results to return format
        std::vector<std::pair<float, long>> results;
        results.reserve(response.result_num);

        for (uint32_t i = 0; i < response.result_num; ++i) {
            results.emplace_back(response.distance[i], static_cast<long>(response.local_idx[i]));
        }

        return results;

    } catch (const std::exception& e) {
        throw std::runtime_error("Error querying Puck index: " + std::string(e.what()));
    }
}

} // namespace puck_index_service
} // namespace knn_jni