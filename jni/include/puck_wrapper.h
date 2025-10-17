/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_PUCK_WRAPPER_H
#define OPENSEARCH_KNN_PUCK_WRAPPER_H

#include <jni.h>
#include <unordered_map>
#include <string>

#include "jni_util.h"

namespace knn_jni {
namespace puck_wrapper {

/**
 * Train a Puck model from training vectors
 * Returns serialized trained model (codebooks)
 * 
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param parametersJ Java Map of training parameters
 * @param dimensionJ Vector dimension
 * @param trainVectorsPointerJ Pointer to training vector data
 * @return jbyteArray containing serialized trained model
 */
jbyteArray TrainIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env,
                      jobject parametersJ, jint dimensionJ, 
                      jlong trainVectorsPointerJ);

/**
 * Build index from trained model template
 * Creates complete searchable index
 * 
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param idsJ Array of document IDs
 * @param vectorsAddressJ Pointer to vector data
 * @param dimJ Vector dimension
 * @param output Output stream for serialized index
 * @param templateIndexJ Trained model template
 * @param parametersJ Build parameters
 */
void CreateIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env,
                             jintArray idsJ, jlong vectorsAddressJ, jint dimJ,
                             jobject output, jbyteArray templateIndexJ,
                             jobject parametersJ);

/**
 * Load index from stream for searching
 * 
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param input Input stream containing serialized index
 * @return Index pointer (jlong)
 */
jlong LoadIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject input);

/**
 * Query index for k-nearest neighbors
 * 
 * @param indexPointer Pointer to loaded index
 * @param queryVector Query vector
 * @param dimension Vector dimension
 * @param k Number of neighbors to find
 * @param parameters Search parameters
 * @param distances Output array for distances
 * @param indices Output array for document IDs
 * @return Number of results found
 */
int knn_query_index(jlong indexPointer, jfloat* queryVector, jint dimension, jint k,
                    const std::unordered_map<std::string, jobject>& parameters,
                    jfloat* distances, jlong* indices);

/**
 * Free index from memory
 * 
 * @param indexPointer Pointer to index to free
 */
void knn_free_index(jlong indexPointer);

jobjectArray QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ);

// ========== DEPRECATED / STUB METHODS ==========
// These throw exceptions - use the above methods instead

/**
 * @deprecated Use TrainIndex + CreateIndexFromTemplate instead
 */
jlong knn_new_index(jlong numDocs, jint dimension,
                   const std::unordered_map<std::string, jobject>& parameters);

/**
 * @deprecated Use TrainIndex + CreateIndexFromTemplate instead
 */
void CreateIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env,
                jintArray idsJ, jlong vectorsAddressJ, jint dimJ,
                jobject output, jobject parametersJ);

} // namespace puck_wrapper
} // namespace knn_jni

#endif // OPENSEARCH_KNN_PUCK_WRAPPER_H
