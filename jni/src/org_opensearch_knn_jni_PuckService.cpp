/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "org_opensearch_knn_jni_PuckService.h"

#include <jni.h>
#include <string>

#include "puck_wrapper.h"
#include "jni_util.h"
#include <memory>

static knn_jni::JNIUtil jniUtil;

// Initialize library
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_PuckService_initLibrary(JNIEnv * env, jclass cls)
{
    jniUtil.Initialize(env);
}

// Create index
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_PuckService_createIndex(
    JNIEnv *env, jclass cls, jintArray idsJ, jlong vectorsAddressJ, jint dimJ,
    jobject output, jobject parametersJ) {
    try {
        knn_jni::puck_wrapper::CreateIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, output, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

// Initialize index
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_PuckService_initIndex(JNIEnv *env, jclass cls, jlong numDocs, jint dimension, jobject parameters) {
    try {
        // Convert Java Map to C++ unordered_map
        std::unordered_map<std::string, jobject> parametersMap;
        if (parameters != nullptr) {
            parametersMap = jniUtil.ConvertJavaMapToCppMap(env, parameters);
        }

        // Create new Puck index
        return knn_jni::puck_wrapper::knn_new_index(numDocs, dimension, parametersMap);

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
        return 0;
    }
}

// Train index and return byte array (NEW API matching FAISS)
JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_PuckService_trainIndex(JNIEnv *env, jclass cls,
                                                                                jobject parametersJ,
                                                                                jint dimensionJ,
                                                                                jlong trainVectorsPointerJ) {
    try {
        // Validate parameters before calling wrapper
        if (parametersJ == nullptr) {
            jniUtil.ThrowJavaException(env, "Parameters object is null");
            return nullptr;
        }
        if (trainVectorsPointerJ == 0) {
            jniUtil.ThrowJavaException(env, "Training vectors pointer is null (0)");
            return nullptr;
        }
        if (dimensionJ <= 0) {
            jniUtil.ThrowJavaException(env, "Invalid dimension value");
            return nullptr;
        }

        return knn_jni::puck_wrapper::TrainIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

// Query index for k-nearest neighbors
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_PuckService_queryIndex(JNIEnv *env, jclass cls, jlong indexPointer, jfloatArray queryVector, jint k, jobject parameters) {
    try {
        if (indexPointer == 0) {
            throw std::invalid_argument("Index pointer is null");
        }

        // Get query vector from Java array
        jfloat* query = env->GetFloatArrayElements(queryVector, nullptr);
        if (query == nullptr) {
            throw std::runtime_error("Failed to get query vector");
        }

        jint dimension = env->GetArrayLength(queryVector);

        // Convert Java Map to C++ unordered_map
        std::unordered_map<std::string, jobject> parametersMap;
        if (parameters != nullptr) {
            parametersMap = jniUtil.ConvertJavaMapToCppMap(env, parameters);
        }

        // Allocate arrays for results
        auto distances = std::make_unique<jfloat[]>(k);
        auto indices = std::make_unique<jlong[]>(k);

        // Query the index
        int resultCount = knn_jni::puck_wrapper::knn_query_index(
            indexPointer, query, dimension, k, parametersMap,
            distances.get(), indices.get()
        );

        // Release query vector
        env->ReleaseFloatArrayElements(queryVector, query, JNI_ABORT);

        // Convert results to KNNQueryResult array
        jclass knnQueryResultClass = env->FindClass("org/opensearch/knn/index/query/KNNQueryResult");
        if (knnQueryResultClass == nullptr) {
            throw std::runtime_error("Failed to find KNNQueryResult class");
        }

        jmethodID constructor = env->GetMethodID(knnQueryResultClass, "<init>", "(IF)V");
        if (constructor == nullptr) {
            throw std::runtime_error("Failed to find KNNQueryResult constructor");
        }

        jobjectArray results = env->NewObjectArray(resultCount, knnQueryResultClass, nullptr);
        if (results == nullptr) {
            throw std::runtime_error("Failed to create result array");
        }

        for (int i = 0; i < resultCount; ++i) {
            jobject result = env->NewObject(knnQueryResultClass, constructor, (jint)indices[i], (jfloat)distances[i]);
            if (result != nullptr) {
                env->SetObjectArrayElement(results, i, result);
                env->DeleteLocalRef(result);
            }
        }

        return results;

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
        return nullptr;
    }
}

// Load index from stream
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_PuckService_loadIndexWithStream(
    JNIEnv *env, jclass cls, jobject input) {
    try {
        return knn_jni::puck_wrapper::LoadIndex(&jniUtil, env, input);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
        return 0;
    }
}

// Load binary index from file path (Puck doesn't distinguish binary, but keeping for API compatibility)
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_PuckService_loadBinaryIndex(
    JNIEnv *env, jclass cls, jstring indexPath) {
    try {
        const char* path = env->GetStringUTFChars(indexPath, nullptr);
        if (path == nullptr) {
            throw std::runtime_error("Failed to get index path");
        }

        // Puck uses file-based loading - this would need proper implementation
        // For now, throw an exception as we primarily use stream-based loading
        env->ReleaseStringUTFChars(indexPath, path);
        throw std::runtime_error("File-based loading not yet implemented for Puck. Use stream-based loading.");

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
        return 0;
    }
}

// Load binary index from stream (Puck doesn't distinguish binary, delegates to loadIndexWithStream)
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_PuckService_loadBinaryIndexWithStream(
    JNIEnv *env, jclass cls, jobject input) {
    return Java_org_opensearch_knn_jni_PuckService_loadIndexWithStream(env, cls, input);
}

// Load index from file path
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_PuckService_loadIndex(
    JNIEnv *env, jclass cls, jstring indexPath) {
    return Java_org_opensearch_knn_jni_PuckService_loadBinaryIndex(env, cls, indexPath);
}

// Free index from memory
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_PuckService_freeIndex(JNIEnv *env, jclass cls, jlong indexPointer) {
    try {
        if (indexPointer != 0) {
            knn_jni::puck_wrapper::knn_free_index(indexPointer);
        }
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

// Create index from trained template
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_PuckService_createIndexFromTemplate(
    JNIEnv *env, jclass cls, jintArray idsJ, jlong vectorsAddressJ, jint dimJ,
    jobject output, jbyteArray templateIndexJ, jobject parametersJ) {
    try {
        knn_jni::puck_wrapper::CreateIndexFromTemplate(&jniUtil, env, idsJ, vectorsAddressJ, dimJ,
                                                       output, templateIndexJ, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}
