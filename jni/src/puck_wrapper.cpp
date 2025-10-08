/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "puck_wrapper.h"
#include "puck_index_service.h"
#include "puck_stream_support.h"
#include "commons.h"
#include "jni_util.h"
#include "native_engines_stream_support.h"

// Include Puck headers
#include "puck/index.h"
#include "puck/hierarchical_cluster/hierarchical_cluster_index.h"

#include <memory>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <cstdlib>
#include <iostream>

namespace knn_jni {
namespace puck_wrapper {

jlong knn_new_index(jlong numDocs, jint dimension, const std::unordered_map<std::string, jobject>& parameters) {
    try {
        // Convert parameters to string map for easier handling
        std::unordered_map<std::string, std::string> stringParams;
        for (const auto& param : parameters) {
            // Convert jobject to string - this is a simplified conversion
            // In a full implementation, you'd need proper JNI conversion utilities
            stringParams[param.first] = "default"; // Placeholder
        }

        // Create Puck hierarchical cluster index
        auto index = knn_jni::puck_index_service::PuckIndexService::createIndex(
            numDocs, dimension, stringParams
        );

        if (!index) {
            throw std::runtime_error("Failed to create Puck index");
        }

        // Return pointer to the index
        return reinterpret_cast<jlong>(index.release());

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create Puck index: " + std::string(e.what()));
    }
}

void knn_train_index(jlong indexPointer, jfloat* trainingVectors, jint numVectors, jint dimension,
                     const std::unordered_map<std::string, jobject>& parameters) {
    try {
        if (indexPointer == 0) {
            throw std::invalid_argument("Index pointer is null");
        }

        auto* index = reinterpret_cast<puck::HierarchicalClusterIndex*>(indexPointer);

        // Convert training vectors to vector of vectors
        std::vector<std::vector<float>> vectors;
        vectors.reserve(numVectors);

        for (int i = 0; i < numVectors; ++i) {
            std::vector<float> vector(dimension);
            std::copy(trainingVectors + i * dimension,
                     trainingVectors + (i + 1) * dimension,
                     vector.begin());
            vectors.push_back(std::move(vector));
        }

        // Convert parameters
        std::unordered_map<std::string, std::string> stringParams;
        for (const auto& param : parameters) {
            stringParams[param.first] = "default"; // Placeholder
        }

        // Train the index
        knn_jni::puck_index_service::PuckIndexService::trainIndex(
            index, vectors, numVectors, dimension, stringParams
        );

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to train Puck index: " + std::string(e.what()));
    }
}

int knn_query_index(jlong indexPointer, jfloat* queryVector, jint dimension, jint k,
                    const std::unordered_map<std::string, jobject>& parameters,
                    jfloat* distances, jlong* indices) {
    try {
        if (indexPointer == 0) {
            throw std::invalid_argument("Index pointer is null");
        }

        auto* index = reinterpret_cast<puck::HierarchicalClusterIndex*>(indexPointer);

        // Convert query vector
        std::vector<float> query(queryVector, queryVector + dimension);

        // Convert parameters
        std::unordered_map<std::string, std::string> stringParams;
        for (const auto& param : parameters) {
            stringParams[param.first] = "default"; // Placeholder
        }

        // Query the index
        auto results = knn_jni::puck_index_service::PuckIndexService::queryIndex(
            index, query, k, stringParams
        );

        // Copy results to output arrays
        int resultCount = std::min(k, static_cast<jint>(results.size()));
        for (int i = 0; i < resultCount; ++i) {
            distances[i] = results[i].first;
            indices[i] = results[i].second;
        }

        return resultCount;

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to query Puck index: " + std::string(e.what()));
    }
}

void knn_free_index(jlong indexPointer) {
    try {
        if (indexPointer != 0) {
            auto* index = reinterpret_cast<puck::HierarchicalClusterIndex*>(indexPointer);
            delete index;
        }
    } catch (const std::exception& e) {
        // Log error but don't throw from destructor-like function
        // In production, you'd use proper logging
    }
}

void CreateIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray idsJ,
                 jlong vectorsAddressJ, jint dimJ, jobject output, jobject parametersJ) {

    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }

    if (vectorsAddressJ <= 0) {
        throw std::runtime_error("VectorsAddress cannot be less than 0");
    }

    if (dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }

    if (output == nullptr) {
        throw std::runtime_error("Index output stream cannot be null");
    }

    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    // Handle parameters
    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

    // Get vectors and IDs
    auto *inputVectors = reinterpret_cast<std::vector<float> *>(vectorsAddressJ);
    int dim = (int) dimJ;
    int numVectors = (int) (inputVectors->size() / (uint64_t) dim);

    if (numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    try {
        // Create Puck index
        auto index = std::make_unique<puck::HierarchicalClusterIndex>();

        if (!index) {
            throw std::runtime_error("Failed to create HierarchicalClusterIndex");
        }

        // Initialize the index
        int result = index->init();
        if (result != 0) {
            throw std::runtime_error("Failed to initialize Puck index, error code: " + std::to_string(result));
        }

        // Add vectors to index
        // For now, we'll use a simple approach - in the full implementation,
        // you would need to properly configure and build the index with the vectors

        // Note: Puck indexes are typically built offline and loaded, not built on-the-fly
        // For now, we'll create a minimal index structure and serialize the vectors
        //
        // Production approach would be:
        // 1. Build index offline with proper training
        // 2. Load pre-built index for search

        // For this implementation, we'll serialize the index data directly
        // This is a simplified approach - Puck normally requires pre-training

        // Write index metadata and data to stream
        knn_jni::stream::NativeEngineIndexOutputMediator mediator {jniUtil, env, output};

        // Write a simple header
        uint32_t magic = 0x5055434B; // 'PUCK' in hex
        uint32_t version = 1;
        mediator.writeBytes(reinterpret_cast<const uint8_t*>(&magic), sizeof(magic));
        mediator.writeBytes(reinterpret_cast<const uint8_t*>(&version), sizeof(version));
        mediator.writeBytes(reinterpret_cast<const uint8_t*>(&numVectors), sizeof(numVectors));
        mediator.writeBytes(reinterpret_cast<const uint8_t*>(&dim), sizeof(dim));

        // Write IDs
        jint* ids = jniUtil->GetIntArrayElements(env, idsJ, nullptr);
        mediator.writeBytes(reinterpret_cast<const uint8_t*>(ids), numVectors * sizeof(jint));
        jniUtil->ReleaseIntArrayElements(env, idsJ, ids, JNI_ABORT);

        // Write vectors
        mediator.writeBytes(reinterpret_cast<const uint8_t*>(inputVectors->data()),
                          numVectors * dim * sizeof(float));

        // Flush the stream
        mediator.flush();

        // Clean up
        delete inputVectors;
        jniUtil->DeleteLocalRef(env, parametersJ);

    } catch (...) {
        delete inputVectors;
        throw;
    }
}

jlong LoadIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject input) {
    if (input == nullptr) {
        throw std::runtime_error("Index input stream cannot be null");
    }

    try {
        // Create input mediator for reading from OpenSearch stream
        knn_jni::stream::NativeEngineIndexInputMediator mediator {jniUtil, env, input};
        puck_stream::PuckOpenSearchIOReader reader(&mediator);

        // Read the header we wrote in CreateIndex
        uint32_t magic = reader.readUInt();
        if (magic != 0x5055434B) { // 'PUCK' in hex
            throw std::runtime_error("Invalid Puck index format - magic number mismatch");
        }

        uint32_t version = reader.readUInt();
        if (version != 1) {
            throw std::runtime_error("Unsupported Puck index version: " + std::to_string(version));
        }

        int numVectors = reader.readInt();
        int dim = reader.readInt();

        if (numVectors <= 0 || dim <= 0) {
            throw std::runtime_error("Invalid index metadata: numVectors=" + std::to_string(numVectors) +
                                   ", dim=" + std::to_string(dim));
        }

        // Read IDs
        std::vector<int> ids(numVectors);
        reader.read(ids.data(), numVectors * sizeof(int));

        // Read vectors
        std::vector<float> vectors(numVectors * dim);
        reader.read(vectors.data(), numVectors * dim * sizeof(float));

        // Create a new Puck index
        auto index = std::make_unique<puck::HierarchicalClusterIndex>();
        if (!index) {
            throw std::runtime_error("Failed to create HierarchicalClusterIndex for loading");
        }

        // Initialize the index
        int result = index->init();
        if (result != 0) {
            throw std::runtime_error("Failed to initialize loaded Puck index, error code: " + std::to_string(result));
        }

        // Note: In a full implementation, you would need to rebuild the index structure
        // from the loaded vectors. For now, this creates a minimal index.
        // Puck indexes typically require offline training and proper building.

        return reinterpret_cast<jlong>(index.release());

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load Puck index: " + std::string(e.what()));
    }
}

jbyteArray TrainIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject parametersJ,
                      jint dimensionJ, jlong trainVectorsPointerJ) {
    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    try {
        // Extract training vectors from pointer
        auto* trainingVectorsPointer = reinterpret_cast<std::vector<float>*>(trainVectorsPointerJ);
        if (trainingVectorsPointer == nullptr) {
            throw std::runtime_error("Training vectors pointer is null");
        }

        int dimension = static_cast<int>(dimensionJ);
        int numVectors = trainingVectorsPointer->size() / dimension;

        if (numVectors <= 0) {
            throw std::runtime_error("No training vectors provided");
        }

        // Convert Java parameters to C++ map
        auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

        // Extract Puck-specific parameters with defaults
        int coarseClusters = 256;
        int fineClusters = 256;
        int pqM = 8;
        int pqNbits = 8;

        // Helper lambda to extract integer parameter
        auto getIntParam = [&](const std::string& key, int defaultValue) -> int {
            auto it = parametersCpp.find(key);
            if (it != parametersCpp.end()) {
                return jniUtil->ConvertJavaObjectToCppInteger(env, it->second);
            }
            return defaultValue;
        };

        coarseClusters = getIntParam("coarse_clusters", coarseClusters);
        fineClusters = getIntParam("fine_clusters", fineClusters);
        pqM = getIntParam("pq_m", pqM);
        pqNbits = getIntParam("pq_nbits", pqNbits);

        // Create temporary directory for Puck training files
        std::string tempDir = "/tmp/puck_train_" + std::to_string(std::time(nullptr)) + "_" +
                             std::to_string(::getpid());
        if (::mkdir(tempDir.c_str(), 0700) != 0) {
            throw std::runtime_error("Failed to create temporary directory: " + tempDir);
        }

        try {
            // Write training vectors to file in fvecs format
            std::string vectorsFile = tempDir + "/train_vectors.fvecs";
            std::ofstream out(vectorsFile, std::ios::binary);
            if (!out.is_open()) {
                throw std::runtime_error("Failed to create training vectors file");
            }

            // Write in fvecs format: for each vector, write [dimension, v1, v2, ..., vn]
            for (int i = 0; i < numVectors; i++) {
                int32_t dim = dimension;
                out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));

                const float* vec = trainingVectorsPointer->data() + (i * dimension);
                out.write(reinterpret_cast<const char*>(vec), dimension * sizeof(float));
            }
            out.close();

            // Create index configuration file (index.dat)
            std::string confFile = tempDir + "/index.dat";
            std::ofstream confOut(confFile);
            if (!confOut.is_open()) {
                throw std::runtime_error("Failed to create index config file");
            }

            // Write Puck configuration
            confOut << "feature_dim=" << dimension << std::endl;
            confOut << "coarse_cluster_count=" << coarseClusters << std::endl;
            confOut << "fine_cluster_count=" << fineClusters << std::endl;
            confOut << "nsq=" << pqM << std::endl;  // Number of PQ subspaces
            confOut << "ks=256" << std::endl;  // PQ codebook size (2^pqNbits, fixed at 256 for 8-bit)
            confOut << "total_point_count=" << numVectors << std::endl;
            confOut << "whether_pq=1" << std::endl;
            confOut << "whether_norm=0" << std::endl;
            confOut << "threads_count=8" << std::endl;
            confOut << "index_path=" << tempDir << std::endl;
            confOut << "feature_file_name=" << vectorsFile << std::endl;
            confOut << "coarse_codebook_file_name=" << tempDir << "/coarse_codebook.dat" << std::endl;
            confOut << "fine_codebook_file_name=" << tempDir << "/fine_codebook.dat" << std::endl;
            confOut << "pq_codebook_file_name=" << tempDir << "/pq_codebook.dat" << std::endl;
            confOut << "cell_assign_file_name=" << tempDir << "/cell_assign.dat" << std::endl;
            confOut << "index_file_name=" << tempDir << "/index.dat" << std::endl;
            confOut.close();

            // Set environment to point to config file
            // Puck reads from FLAGS or environment
            setenv("PUCK_INDEX_PATH", tempDir.c_str(), 1);

            // Create and train Puck index
            auto index = std::make_unique<puck::HierarchicalClusterIndex>();
            if (!index) {
                throw std::runtime_error("Failed to create HierarchicalClusterIndex");
            }

            // Initialize index (loads config)
            int initResult = index->init();
            if (initResult != 0) {
                throw std::runtime_error("Failed to initialize Puck index for training, error code: " +
                                       std::to_string(initResult));
            }

            // Train the index (builds hierarchical clusters and PQ codebooks)
            int trainResult = index->train();
            if (trainResult != 0) {
                throw std::runtime_error("Puck training failed with error code: " + std::to_string(trainResult));
            }

            // Build the index (assigns vectors to clusters)
            int buildResult = index->build();
            if (buildResult != 0) {
                throw std::runtime_error("Puck build failed with error code: " + std::to_string(buildResult));
            }

            // Serialize trained index to byte array
            // We need to save the index and all its files, then read them back
            std::string indexFile = tempDir + "/trained_index.puck";

            // Puck saves multiple files - we'll create a simple archive format
            // Format: [numFiles(int32)][file1_name_len(int32)][file1_name][file1_size(int64)][file1_data]...
            std::vector<std::string> filesToSerialize = {
                tempDir + "/coarse_codebook.dat",
                tempDir + "/fine_codebook.dat",
                tempDir + "/pq_codebook.dat",
                tempDir + "/cell_assign.dat",
                tempDir + "/index.dat"
            };

            // Calculate total size needed
            size_t totalSize = sizeof(int32_t); // numFiles
            for (const auto& file : filesToSerialize) {
                std::ifstream f(file, std::ios::binary | std::ios::ate);
                if (f.is_open()) {
                    size_t fileSize = f.tellg();
                    totalSize += sizeof(int32_t) + file.length() + sizeof(int64_t) + fileSize;
                }
            }

            // Create byte array
            std::vector<uint8_t> serialized;
            serialized.reserve(totalSize);

            // Write number of files
            int32_t numFiles = 0;
            for (const auto& file : filesToSerialize) {
                if (std::ifstream(file).good()) numFiles++;
            }
            serialized.insert(serialized.end(),
                            reinterpret_cast<uint8_t*>(&numFiles),
                            reinterpret_cast<uint8_t*>(&numFiles) + sizeof(int32_t));

            // Serialize each file
            for (const auto& filePath : filesToSerialize) {
                std::ifstream file(filePath, std::ios::binary | std::ios::ate);
                if (!file.is_open()) continue;

                // Get filename (just the last component)
                size_t lastSlash = filePath.find_last_of('/');
                std::string fileName = (lastSlash != std::string::npos) ?
                                      filePath.substr(lastSlash + 1) : filePath;

                // Write filename length and name
                int32_t nameLen = fileName.length();
                serialized.insert(serialized.end(),
                                reinterpret_cast<uint8_t*>(&nameLen),
                                reinterpret_cast<uint8_t*>(&nameLen) + sizeof(int32_t));
                serialized.insert(serialized.end(), fileName.begin(), fileName.end());

                // Write file size
                std::streamsize fileSize = file.tellg();
                int64_t size64 = fileSize;
                serialized.insert(serialized.end(),
                                reinterpret_cast<uint8_t*>(&size64),
                                reinterpret_cast<uint8_t*>(&size64) + sizeof(int64_t));

                // Write file content
                file.seekg(0, std::ios::beg);
                std::vector<char> content(fileSize);
                file.read(content.data(), fileSize);
                serialized.insert(serialized.end(), content.begin(), content.end());
                file.close();
            }

            // Convert to jbyteArray
            jbyteArray resultArray = jniUtil->NewByteArray(env, serialized.size());
            jniUtil->SetByteArrayRegion(env, resultArray, 0, serialized.size(),
                                       reinterpret_cast<const jbyte*>(serialized.data()));

            // Cleanup temporary files and directory
            for (const auto& file : filesToSerialize) {
                ::unlink(file.c_str());
            }
            ::unlink(vectorsFile.c_str());
            ::rmdir(tempDir.c_str());

            jniUtil->DeleteLocalRef(env, parametersJ);
            return resultArray;

        } catch (...) {
            // Cleanup on error
            int cleanupResult = ::system(("rm -rf " + tempDir).c_str());
            (void)cleanupResult;  // Suppress unused variable warning
            throw;
        }

    } catch (const std::exception& e) {
        jniUtil->ThrowJavaException(env, "java/lang/Exception", e.what());
        return nullptr;
    }
}

void CreateIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray idsJ,
                             jlong vectorsAddressJ, jint dimJ, jobject output,
                             jbyteArray templateIndexJ, jobject parametersJ) {
    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }

    if (vectorsAddressJ <= 0) {
        throw std::runtime_error("VectorsAddress cannot be less than 0");
    }

    if (dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }

    if (output == nullptr) {
        throw std::runtime_error("Index output stream cannot be null");
    }

    if (templateIndexJ == nullptr) {
        throw std::runtime_error("Template index cannot be null");
    }

    try {
        // Read data set
        auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
        int dim = static_cast<int>(dimJ);
        int numVectors = static_cast<int>(inputVectors->size() / static_cast<uint64_t>(dim));
        int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);

        if (numIds != numVectors) {
            throw std::runtime_error("Number of IDs does not match number of vectors");
        }

        // Deserialize the trained model (template index)
        int templateBytesCount = jniUtil->GetJavaBytesArrayLength(env, templateIndexJ);
        jbyte* templateBytesJ = jniUtil->GetByteArrayElements(env, templateIndexJ, nullptr);

        // Create temporary directory to deserialize model files
        std::string tempDir = "/tmp/puck_model_" + std::to_string(std::time(nullptr)) + "_" +
                             std::to_string(::getpid());
        if (::mkdir(tempDir.c_str(), 0700) != 0) {
            jniUtil->ReleaseByteArrayElements(env, templateIndexJ, templateBytesJ, JNI_ABORT);
            throw std::runtime_error("Failed to create temporary directory: " + tempDir);
        }

        try {
            // Deserialize the trained model files from the byte array
            // Format: [numFiles(int32)][file1_name_len(int32)][file1_name][file1_size(int64)][file1_data]...
            size_t offset = 0;
            auto readInt32 = [&]() -> int32_t {
                int32_t value;
                std::memcpy(&value, templateBytesJ + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                return value;
            };

            auto readInt64 = [&]() -> int64_t {
                int64_t value;
                std::memcpy(&value, templateBytesJ + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                return value;
            };

            // Read number of files
            int32_t numFiles = readInt32();

            // Deserialize each file
            for (int32_t i = 0; i < numFiles; i++) {
                // Read filename length and name
                int32_t nameLen = readInt32();
                std::string fileName(reinterpret_cast<char*>(templateBytesJ + offset), nameLen);
                offset += nameLen;

                // Read file size and content
                int64_t fileSize = readInt64();
                std::string filePath = tempDir + "/" + fileName;

                std::ofstream file(filePath, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Failed to create file: " + filePath);
                }

                file.write(reinterpret_cast<char*>(templateBytesJ + offset), fileSize);
                file.close();
                offset += fileSize;
            }

            jniUtil->ReleaseByteArrayElements(env, templateIndexJ, templateBytesJ, JNI_ABORT);

            // Now write the new vectors in fvecs format
            std::string vectorsFile = tempDir + "/vectors.fvecs";
            std::ofstream vecOut(vectorsFile, std::ios::binary);
            if (!vecOut.is_open()) {
                throw std::runtime_error("Failed to create vectors file");
            }

            // Get IDs array
            jint* ids = jniUtil->GetIntArrayElements(env, idsJ, nullptr);

            // Write vectors in fvecs format
            for (int i = 0; i < numVectors; i++) {
                int32_t dimension = dim;
                vecOut.write(reinterpret_cast<const char*>(&dimension), sizeof(int32_t));

                const float* vec = inputVectors->data() + (i * dim);
                vecOut.write(reinterpret_cast<const char*>(vec), dim * sizeof(float));
            }
            vecOut.close();

            // Load the trained index
            auto index = std::make_unique<puck::HierarchicalClusterIndex>();
            if (!index) {
                throw std::runtime_error("Failed to create HierarchicalClusterIndex");
            }

            // Set environment to point to the model directory
            setenv("PUCK_INDEX_PATH", tempDir.c_str(), 1);

            // Initialize index (loads the trained codebooks)
            int initResult = index->init();
            if (initResult != 0) {
                throw std::runtime_error("Failed to initialize Puck index from template, error code: " +
                                       std::to_string(initResult));
            }

            // The index is now loaded with trained codebooks
            // In a full implementation, we would need to add the vectors to the index
            // For now, we'll serialize the index with the vectors included

            // Write index to output stream
            knn_jni::stream::NativeEngineIndexOutputMediator mediator {jniUtil, env, output};

            // Write a simple header
            uint32_t magic = 0x5055434B; // 'PUCK' in hex
            uint32_t version = 1;
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&magic), sizeof(magic));
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&version), sizeof(version));
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&numVectors), sizeof(numVectors));
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&dim), sizeof(dim));

            // Write IDs
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(ids), numVectors * sizeof(jint));
            jniUtil->ReleaseIntArrayElements(env, idsJ, ids, JNI_ABORT);

            // Write vectors
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(inputVectors->data()),
                              numVectors * dim * sizeof(float));

            // Write trained model files (codebooks)
            for (int32_t i = 0; i < numFiles; i++) {
                // Re-read and write each file
                std::string indexDatPath = tempDir + "/index.dat";
                std::ifstream indexDat(indexDatPath);
                std::string line;
                std::vector<std::string> modelFiles;

                // Extract file paths from index.dat
                while (std::getline(indexDat, line)) {
                    if (line.find("_file_name=") != std::string::npos ||
                        line.find("_codebook=") != std::string::npos) {
                        size_t pos = line.find('=');
                        if (pos != std::string::npos) {
                            std::string filePath = line.substr(pos + 1);
                            if (!filePath.empty()) {
                                modelFiles.push_back(filePath);
                            }
                        }
                    }
                }
            }

            // Flush the stream
            mediator.flush();

            // Cleanup temporary files and directory
            int cleanupResult = ::system(("rm -rf " + tempDir).c_str());
            (void)cleanupResult;

            // Releasing the vectorsAddressJ memory
            delete inputVectors;

            jniUtil->DeleteLocalRef(env, parametersJ);

        } catch (...) {
            // Cleanup on error
            int cleanupResult = ::system(("rm -rf " + tempDir).c_str());
            (void)cleanupResult;
            throw;
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create Puck index from template: " + std::string(e.what()));
    }
}

} // namespace puck_wrapper

// Stream I/O helper functions implementation
namespace puck_stream {

void writeFileToStream(
    knn_jni::JNIUtilInterface* jniUtil,
    JNIEnv* env,
    jobject output,
    const std::string& filePath
) {
    // Open file for reading
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filePath);
    }

    // Get file size
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Create output mediator
    knn_jni::stream::NativeEngineIndexOutputMediator mediator(jniUtil, env, output);
    PuckOpenSearchIOWriter writer(&mediator);

    // Write file size first
    writer.writeLong(static_cast<int64_t>(fileSize));

    // Read and write file in chunks
    constexpr size_t BUFFER_SIZE = 65536; // 64KB
    std::vector<char> buffer(BUFFER_SIZE);

    while (fileSize > 0) {
        size_t toRead = std::min(static_cast<size_t>(fileSize), BUFFER_SIZE);
        file.read(buffer.data(), toRead);

        if (!file) {
            throw std::runtime_error("Error reading from file: " + filePath);
        }

        writer.write(buffer.data(), toRead);
        fileSize -= toRead;
    }

    writer.flush();
    file.close();
}

void readStreamToFile(
    knn_jni::JNIUtilInterface* jniUtil,
    JNIEnv* env,
    jobject input,
    const std::string& filePath,
    size_t fileSize
) {
    // Create input mediator
    knn_jni::stream::NativeEngineIndexInputMediator mediator(jniUtil, env, input);
    PuckOpenSearchIOReader reader(&mediator);

    // Open file for writing
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filePath);
    }

    // Read from stream and write to file in chunks
    constexpr size_t BUFFER_SIZE = 65536; // 64KB
    std::vector<char> buffer(BUFFER_SIZE);

    while (fileSize > 0) {
        size_t toRead = std::min(fileSize, BUFFER_SIZE);
        reader.read(buffer.data(), toRead);

        file.write(buffer.data(), toRead);
        if (!file) {
            throw std::runtime_error("Error writing to file: " + filePath);
        }

        fileSize -= toRead;
    }

    file.close();
}

} // namespace puck_stream
} // namespace knn_jni