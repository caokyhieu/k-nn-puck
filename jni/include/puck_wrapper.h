/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_PUCK_WRAPPER_H
#define OPENSEARCH_KNN_PUCK_WRAPPER_H

#include <jni.h>
#include <memory>
#include <string>
#include <unordered_map>

// Forward declaration
namespace knn_jni {
    class JNIUtilInterface;
}

namespace knn_jni {
namespace puck_wrapper {

/**
 * Create and persist a Puck index from vectors
 *
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param ids Array of document IDs
 * @param vectorsAddress Address of native memory where vectors are stored
 * @param dimension Vector dimension
 * @param output Index output stream for persisting the index
 * @param parameters Index parameters including cluster configuration
 */
void CreateIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray ids,
                 jlong vectorsAddress, jint dimension, jobject output, jobject parameters);

/**
 * Initialize a Puck index with hierarchical clustering
 *
 * @param numDocs Number of documents expected
 * @param dimension Vector dimension
 * @param parameters Index parameters including cluster configuration
 * @return Pointer to the created index
 */
jlong knn_new_index(jlong numDocs, jint dimension, const std::unordered_map<std::string, jobject>& parameters);

/**
 * Train the Puck index with hierarchical clustering
 *
 * @param indexPointer Pointer to the index
 * @param trainingVectors Training data vectors
 * @param numVectors Number of training vectors
 * @param dimension Vector dimension
 * @param parameters Training parameters
 */
void knn_train_index(jlong indexPointer, jfloat* trainingVectors, jint numVectors, jint dimension,
                     const std::unordered_map<std::string, jobject>& parameters);

/**
 * Query the Puck index for k nearest neighbors
 *
 * @param indexPointer Pointer to the index
 * @param queryVector Query vector
 * @param dimension Vector dimension
 * @param k Number of nearest neighbors to return
 * @param parameters Query parameters
 * @param distances Output distances array
 * @param indices Output indices array
 * @return Number of results found
 */
int knn_query_index(jlong indexPointer, jfloat* queryVector, jint dimension, jint k,
                    const std::unordered_map<std::string, jobject>& parameters,
                    jfloat* distances, jlong* indices);

/**
 * Load a Puck index from a stream
 *
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param input Index input stream for reading the persisted index
 * @return Pointer to the loaded index
 */
jlong LoadIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject input);

/**
 * Free the Puck index from memory
 *
 * @param indexPointer Pointer to the index to be freed
 */
void knn_free_index(jlong indexPointer);

/**
 * Train a Puck index and return as byte array (matches FAISS API)
 *
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param parametersJ Index parameters (coarse_clusters, fine_clusters, pq_m, pq_nbits)
 * @param dimensionJ Vector dimension
 * @param trainVectorsPointerJ Pointer to training vectors in native memory
 * @return Trained index serialized as byte array
 */
jbyteArray TrainIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject parametersJ,
                      jint dimensionJ, jlong trainVectorsPointerJ);

/**
 * Create index from a trained template model (matches FAISS API)
 *
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param ids Array of document IDs
 * @param vectorsAddress Address of native memory where vectors are stored
 * @param dimension Vector dimension
 * @param output Index output stream for persisting the index
 * @param templateIndex Trained model blob (from trainIndex)
 * @param parameters Additional parameters
 */
void CreateIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray ids,
                             jlong vectorsAddress, jint dimension, jobject output,
                             jbyteArray templateIndex, jobject parameters);

} // namespace puck_wrapper
} // namespace knn_jni

#endif // OPENSEARCH_KNN_PUCK_WRAPPER_H