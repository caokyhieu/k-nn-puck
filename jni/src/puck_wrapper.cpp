/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "puck_wrapper.h"
#include "jni_util.h"
#include "native_engines_stream_support.h"
#include "puck_stream_support.h"

#include <vector>
#include <memory>
#include <cstring>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <cerrno>

#include "puck/puck/puck_index.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/search_context.h"

namespace {
    std::string createUniqueTempDir(const std::string& prefix) {
        std::string templateStr = "/tmp/" + prefix + "_XXXXXX";
        std::vector<char> template_chars(templateStr.begin(), templateStr.end());
        template_chars.push_back('\0');
        
        char* result = ::mkdtemp(template_chars.data());
        if (result == nullptr) {
            throw std::runtime_error("Failed to create temp dir: " + std::string(std::strerror(errno)));
        }
        
        return std::string(result);
    }

    puck::IndexConf readIndexConf(const std::string& indexDatPath) {
        std::ifstream file(indexDatPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open index.dat");
        }

        puck::IndexConf conf;
        
        size_t configSize;
        file.read(reinterpret_cast<char*>(&configSize), sizeof(size_t));
        
        uint32_t indexType;
        file.read(reinterpret_cast<char*>(&indexType), sizeof(uint32_t));
        conf.index_type = static_cast<puck::IndexType>(indexType);
        
        file.read(reinterpret_cast<char*>(&conf.feature_dim), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&conf.whether_norm), sizeof(bool));
        file.read(reinterpret_cast<char*>(&conf.total_point_count), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&conf.whether_filter), sizeof(bool));
        file.read(reinterpret_cast<char*>(&conf.filter_nsq), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&conf.ks), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&conf.whether_pq), sizeof(bool));
        file.read(reinterpret_cast<char*>(&conf.nsq), sizeof(uint32_t));
        
        uint32_t ks2;
        file.read(reinterpret_cast<char*>(&ks2), sizeof(uint32_t));
        
        file.read(reinterpret_cast<char*>(&conf.ip2cos), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&conf.coarse_cluster_count), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&conf.fine_cluster_count), sizeof(uint32_t));
        
        file.close();
        
        conf.topk = 100;
        conf.search_coarse_count = 200;
        conf.neighbors_count = 40000;
        conf.filter_topk = 1100;
        conf.radius_rate = 1.0;
        
        return conf;
    }
}

class PuckIndexWrapper : public puck::PuckIndex {
public:
    int public_init_single_build() {
        return init_single_build();
    }
    
    int public_single_build(puck::PuckBuildInfo* build_info) {
        return single_build(build_info);
    }
    
    void public_set_conf(const puck::IndexConf& conf) {
        _conf = conf;
    }
    
    int public_init_model_memory() {
        return init_model_memory();
    }
    
    int public_read_coodbooks() {
        return read_coodbooks();
    }
    
    int public_convert_local_to_memory_idx(uint32_t* cell_start_memory_idx, 
                                           uint32_t* local_to_memory_idx) {
        return convert_local_to_memory_idx(cell_start_memory_idx, local_to_memory_idx);
    }
    
    int public_read_feature_index(uint32_t* local_to_memory_idx) {
        return read_feature_index(local_to_memory_idx);
    }
    
    int public_search(puck::Request* request, puck::Response* response) {
        return search(request, response);
    }
    
    puck::IndexConf& get_index_conf() {
        return _conf;
    }
};

namespace knn_jni {
namespace puck_wrapper {

struct IndexMetadata {
    std::vector<int> ids;
    std::vector<float> vectors;
    std::vector<uint32_t> cellAssignments;
    uint32_t dimension;
    uint32_t numVectors;
    std::unique_ptr<PuckIndexWrapper> puckIndex;
    puck::IndexConf indexConf;
    std::string tempDir;  // ✅ ADD THIS

    ~IndexMetadata() {
        // Clean up temp directory
        if (!tempDir.empty()) {
            ::system(("rm -rf " + tempDir).c_str());
        }
    }
};



// Train Index
jbyteArray TrainIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject parametersJ,
                      jint dimensionJ, jlong trainVectorsPointerJ) {
    if (parametersJ == nullptr || trainVectorsPointerJ == 0) {
        throw std::runtime_error("Invalid parameters");
    }

    try {
        auto* trainingVectors = reinterpret_cast<std::vector<float>*>(trainVectorsPointerJ);
        int dimension = static_cast<int>(dimensionJ);
        int numVectors = trainingVectors->size() / dimension;

        std::cerr << "[Puck] Training with " << numVectors << " vectors, dim=" << dimension << std::endl;

        auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);
        
        auto getIntParam = [&](const std::string& key, int defaultValue) -> int {
            auto it = parametersCpp.find(key);
            if (it != parametersCpp.end()) {
                return jniUtil->ConvertJavaObjectToCppInteger(env, it->second);
            }
            return defaultValue;
        };

        int coarseClusters = getIntParam("coarse_clusters", 100);
        int fineClusters = getIntParam("fine_clusters", 100);
        int pqM = getIntParam("pq_m", 8);

        std::string tempDir = createUniqueTempDir("puck_train");

        try {
            std::string vectorsFile = tempDir + "/train_vectors.fvecs";
            std::ofstream out(vectorsFile, std::ios::binary);
            if (!out.is_open()) {
                throw std::runtime_error("Failed to create training file");
            }

            for (int i = 0; i < numVectors; i++) {
                int32_t dim = dimension;
                out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
                const float* vec = trainingVectors->data() + (i * dimension);
                out.write(reinterpret_cast<const char*>(vec), dimension * sizeof(float));
            }
            out.close();

            google::SetCommandLineOption("index_path", tempDir.c_str());
            google::SetCommandLineOption("feature_dim", std::to_string(dimension).c_str());
            google::SetCommandLineOption("coarse_cluster_count", std::to_string(coarseClusters).c_str());
            google::SetCommandLineOption("fine_cluster_count", std::to_string(fineClusters).c_str());
            google::SetCommandLineOption("nsq", std::to_string(pqM).c_str());
            google::SetCommandLineOption("filter_nsq", std::to_string(dimension / 4).c_str());
            google::SetCommandLineOption("whether_pq", "true");
            google::SetCommandLineOption("whether_filter", "true");
            google::SetCommandLineOption("whether_norm", "false");
            google::SetCommandLineOption("threads_count", "8");
            google::SetCommandLineOption("feature_file_name", "train_vectors.fvecs");
            google::SetCommandLineOption("coarse_codebook_file_name", "coarse_codebook.dat");
            google::SetCommandLineOption("fine_codebook_file_name", "fine_codebook.dat");
            google::SetCommandLineOption("filter_codebook_file_name", "filter_codebook.dat");
            google::SetCommandLineOption("pq_codebook_file_name", "pq_codebook.dat");
            google::SetCommandLineOption("filter_data_file_name", "filter_data.dat");
            google::SetCommandLineOption("pq_data_file_name", "pq_data.dat");
            google::SetCommandLineOption("index_file_name", "index.dat");

            auto puckIndex = std::make_unique<PuckIndexWrapper>();
            if (!puckIndex) {
                throw std::runtime_error("Failed to create PuckIndex");
            }

            std::cerr << "[Puck] Starting training..." << std::endl;
            int trainResult = puckIndex->train();
            if (trainResult != 0) {
                throw std::runtime_error("Training failed: " + std::to_string(trainResult));
            }
            std::cerr << "[Puck] Training completed" << std::endl;

            // Serialize model files
            std::vector<std::string> modelFiles = {
                "coarse_codebook.dat",
                "fine_codebook.dat",
                "filter_codebook.dat",
                "pq_codebook.dat",
                "index.dat"
            };

            size_t totalSize = sizeof(int32_t);
            for (const auto& filename : modelFiles) {
                std::string filepath = tempDir + "/" + filename;
                std::ifstream f(filepath, std::ios::binary | std::ios::ate);
                if (f.is_open()) {
                    size_t fileSize = f.tellg();
                    totalSize += sizeof(int32_t) + filename.length() + sizeof(int64_t) + fileSize;
                }
            }

            std::vector<uint8_t> serialized;
            serialized.reserve(totalSize);

            int32_t numFiles = 0;
            for (const auto& filename : modelFiles) {
                std::string filepath = tempDir + "/" + filename;
                if (std::ifstream(filepath).good()) numFiles++;
            }
            serialized.insert(serialized.end(),
                            reinterpret_cast<uint8_t*>(&numFiles),
                            reinterpret_cast<uint8_t*>(&numFiles) + sizeof(int32_t));

            for (const auto& filename : modelFiles) {
                std::string filepath = tempDir + "/" + filename;
                std::ifstream file(filepath, std::ios::binary | std::ios::ate);
                if (!file.is_open()) continue;

                int32_t nameLen = filename.length();
                serialized.insert(serialized.end(),
                                reinterpret_cast<uint8_t*>(&nameLen),
                                reinterpret_cast<uint8_t*>(&nameLen) + sizeof(int32_t));
                serialized.insert(serialized.end(), filename.begin(), filename.end());

                std::streamsize fileSize = file.tellg();
                int64_t size64 = fileSize;
                serialized.insert(serialized.end(),
                                reinterpret_cast<uint8_t*>(&size64),
                                reinterpret_cast<uint8_t*>(&size64) + sizeof(int64_t));

                file.seekg(0, std::ios::beg);
                std::vector<char> content(fileSize);
                file.read(content.data(), fileSize);
                serialized.insert(serialized.end(), content.begin(), content.end());
                file.close();
            }

            jbyteArray result = jniUtil->NewByteArray(env, serialized.size());
            jniUtil->SetByteArrayRegion(env, result, 0, serialized.size(),
                                       reinterpret_cast<const jbyte*>(serialized.data()));

            ::system(("rm -rf " + tempDir).c_str());
            jniUtil->DeleteLocalRef(env, parametersJ);

            std::cerr << "[Puck] Training complete, model size=" << serialized.size() << " bytes" << std::endl;
            return result;

        } catch (...) {
            ::system(("rm -rf " + tempDir).c_str());
            throw;
        }

    } catch (const std::exception& e) {
        jniUtil->ThrowJavaException(env, "java/lang/Exception", e.what());
        return nullptr;
    }
}

// CreateIndexFromTemplate
void CreateIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray idsJ,
                             jlong vectorsAddressJ, jint dimJ, jobject output,
                             jbyteArray templateIndexJ, jobject parametersJ) {
    if (idsJ == nullptr || vectorsAddressJ <= 0 || dimJ <= 0 || 
        output == nullptr || templateIndexJ == nullptr) {
        throw std::runtime_error("Invalid parameters");
    }

    try {
        auto* inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
        int dim = static_cast<int>(dimJ);
        int numVectors = inputVectors->size() / dim;
        int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);

        if (numIds != numVectors) {
            throw std::runtime_error("ID count mismatch");
        }

        std::cerr << "[Puck] Creating index: " << numVectors << " vectors, dim=" << dim << std::endl;

        int templateSize = jniUtil->GetJavaBytesArrayLength(env, templateIndexJ);
        jbyte* templateBytes = jniUtil->GetByteArrayElements(env, templateIndexJ, nullptr);

        std::string tempDir = createUniqueTempDir("puck_build");

        try {
            size_t offset = 0;
            auto readInt32 = [&]() -> int32_t {
                int32_t value;
                std::memcpy(&value, templateBytes + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                return value;
            };

            auto readInt64 = [&]() -> int64_t {
                int64_t value;
                std::memcpy(&value, templateBytes + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                return value;
            };

            int32_t numFiles = readInt32();
            std::vector<std::string> fileNames;

            std::cerr << "[Puck] Deserializing " << numFiles << " model files..." << std::endl;

            for (int32_t i = 0; i < numFiles; i++) {
                int32_t nameLen = readInt32();
                std::string fileName(reinterpret_cast<char*>(templateBytes + offset), nameLen);
                offset += nameLen;

                int64_t fileSize = readInt64();
                std::string filePath = tempDir + "/" + fileName;

                std::ofstream file(filePath, std::ios::binary);
                file.write(reinterpret_cast<char*>(templateBytes + offset), fileSize);
                file.close();
                offset += fileSize;

                fileNames.push_back(fileName);
                std::cerr << "[Puck]   - " << fileName << " (" << fileSize << " bytes)" << std::endl;
            }

            jniUtil->ReleaseByteArrayElements(env, templateIndexJ, templateBytes, JNI_ABORT);

            // ✅ Read config FIRST
            std::string indexDatPath = tempDir + "/index.dat";
            puck::IndexConf conf = readIndexConf(indexDatPath);

            std::cerr << "[Puck] Config: filter_nsq=" << conf.filter_nsq 
                     << ", nsq=" << conf.nsq << ", ks=" << conf.ks << std::endl;

            // ✅ Calculate sizes from config
            const int filterBytesPerVector = sizeof(float) + conf.filter_nsq;
            const int pqBytesPerVector = sizeof(float) + conf.nsq;

            std::cerr << "[Puck] Quantization format: filter=" << filterBytesPerVector 
                     << " bytes/vec, pq=" << pqBytesPerVector << " bytes/vec" << std::endl;

            // ✅ Set gflags BEFORE init_single_build
            google::SetCommandLineOption("index_path", tempDir.c_str());
            google::SetCommandLineOption("feature_dim", std::to_string(dim).c_str());
            google::SetCommandLineOption("total_point_count", std::to_string(numVectors).c_str());
            google::SetCommandLineOption("filter_data_file_name", "filter_data.dat");
            google::SetCommandLineOption("pq_data_file_name", "pq_data.dat");
            google::SetCommandLineOption("coarse_codebook_file_name", "coarse_codebook.dat");
            google::SetCommandLineOption("fine_codebook_file_name", "fine_codebook.dat");
            google::SetCommandLineOption("filter_codebook_file_name", "filter_codebook.dat");
            google::SetCommandLineOption("pq_codebook_file_name", "pq_codebook.dat");
            google::SetCommandLineOption("index_file_name", "index.dat");
            google::SetCommandLineOption("cell_assign_file_name", "cell_assign.dat");

            auto index = std::make_unique<PuckIndexWrapper>();
            
            std::cerr << "[Puck] Initializing PuckIndex for building..." << std::endl;
            int initResult = index->public_init_single_build();
            if (initResult != 0) {
                throw std::runtime_error("init_single_build failed: " + std::to_string(initResult));
            }

            jint* ids = jniUtil->GetIntArrayElements(env, idsJ, nullptr);

            // Use byte buffers sized from config
            std::vector<uint8_t> filterData(numVectors * filterBytesPerVector);
            std::vector<uint8_t> pqData(numVectors * pqBytesPerVector);
            std::vector<uint32_t> cellAssignments(numVectors);

            std::cerr << "[Puck] Building index with " << numVectors << " vectors..." << std::endl;
            
            for (int i = 0; i < numVectors; i++) {
                puck::PuckBuildInfo buildInfo;
                buildInfo.feature.assign(
                    inputVectors->data() + i * dim,
                    inputVectors->data() + (i + 1) * dim
                );

                int buildResult = index->public_single_build(&buildInfo);
                if (buildResult != 0) {
                    throw std::runtime_error("Build failed for vector " + std::to_string(i));
                }

                cellAssignments[i] = buildInfo.nearest_cell.cell_id;

                // Extract quantization: [offset:4][codes:nsq]
                if (buildInfo.quantizated_feature.size() >= 2) {
                    // Filter quantization
                    auto& filterQuant = buildInfo.quantizated_feature[0];
                    float filterOffset = filterQuant.first;
                    const auto& filterCodes = filterQuant.second;
                    
                    size_t filterPos = i * filterBytesPerVector;
                    std::memcpy(&filterData[filterPos], &filterOffset, sizeof(float));
                    std::memcpy(&filterData[filterPos + sizeof(float)], 
                               filterCodes.data(), 
                               std::min(filterCodes.size(), (size_t)conf.filter_nsq));
                    
                    // PQ quantization
                    auto& pqQuant = buildInfo.quantizated_feature[1];
                    float pqOffset = pqQuant.first;
                    const auto& pqCodes = pqQuant.second;
                    
                    size_t pqPos = i * pqBytesPerVector;
                    std::memcpy(&pqData[pqPos], &pqOffset, sizeof(float));
                    std::memcpy(&pqData[pqPos + sizeof(float)], 
                               pqCodes.data(), 
                               std::min(pqCodes.size(), (size_t)conf.nsq));
                    
                    if (i == 0) {
                        std::cerr << "[Puck] Vector 0 quantization:" << std::endl;
                        std::cerr << "  Filter: offset=" << filterOffset 
                                 << ", codes=" << filterCodes.size() << " bytes" << std::endl;
                        std::cerr << "  PQ: offset=" << pqOffset 
                                 << ", codes=" << pqCodes.size() << " bytes" << std::endl;
                    }
                } else {
                    throw std::runtime_error("Quantization data not generated for vector " + std::to_string(i));
                }

                if ((i + 1) % 1000 == 0) {
                    std::cerr << "[Puck] Built " << (i + 1) << "/" << numVectors << std::endl;
                }
            }

            std::cerr << "[Puck] All vectors processed with single_build" << std::endl;

            // Write quantization data
            std::string filterDataPath = tempDir + "/filter_data.dat";
            std::string pqDataPath = tempDir + "/pq_data.dat";
            std::string cellAssignPath = tempDir + "/cell_assign.dat";

            std::ofstream filterFile(filterDataPath, std::ios::binary);
            filterFile.write(reinterpret_cast<const char*>(filterData.data()), filterData.size());
            filterFile.close();

            std::ofstream pqFile(pqDataPath, std::ios::binary);
            pqFile.write(reinterpret_cast<const char*>(pqData.data()), pqData.size());
            pqFile.close();

            std::ofstream cellFile(cellAssignPath, std::ios::binary);
            cellFile.write(reinterpret_cast<const char*>(cellAssignments.data()), 
                          numVectors * sizeof(uint32_t));
            cellFile.close();

            std::cerr << "[Puck] Quantization data written:" << std::endl;
            std::cerr << "  - filter_data.dat: " << filterData.size() 
                     << " bytes (" << filterBytesPerVector << " bytes/vector)" << std::endl;
            std::cerr << "  - pq_data.dat: " << pqData.size() 
                     << " bytes (" << pqBytesPerVector << " bytes/vector)" << std::endl;
            std::cerr << "  - cell_assign.dat: " << (numVectors * sizeof(uint32_t)) << " bytes" << std::endl;

            std::cerr << "[Puck] Index built successfully" << std::endl;

            // Write to OpenSearch stream
            std::cerr << "[Puck] Serializing index to OpenSearch..." << std::endl;
            stream::NativeEngineIndexOutputMediator mediator(jniUtil, env, output);

            uint32_t magic = 0x5055434B;
            uint32_t version = 1;
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&magic), sizeof(magic));
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&version), sizeof(version));

            uint32_t dimension = dim;
            uint32_t numVectorsU32 = numVectors;
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&dimension), sizeof(dimension));
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&numVectorsU32), sizeof(numVectorsU32));

            mediator.writeBytes(reinterpret_cast<const uint8_t*>(ids), numVectors * sizeof(jint));
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(inputVectors->data()),
                              numVectors * dim * sizeof(float));
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(cellAssignments.data()),
                              numVectors * sizeof(uint32_t));

            std::vector<std::string> allFiles = fileNames;
            allFiles.push_back("filter_data.dat");
            allFiles.push_back("pq_data.dat");
            allFiles.push_back("cell_assign.dat");

            int32_t numAllFiles = allFiles.size();
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&numAllFiles), sizeof(numAllFiles));

            std::cerr << "[Puck] Writing " << numAllFiles << " files to index..." << std::endl;

            for (const auto& filename : allFiles) {
                std::string filepath = tempDir + "/" + filename;
                std::ifstream file(filepath, std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    throw std::runtime_error("Missing required file: " + filename);
                }

                std::streamsize fileSize = file.tellg();
                file.seekg(0, std::ios::beg);

                int32_t nameLen = filename.length();
                mediator.writeBytes(reinterpret_cast<const uint8_t*>(&nameLen), sizeof(nameLen));
                mediator.writeBytes(reinterpret_cast<const uint8_t*>(filename.c_str()), nameLen);

                int64_t fileSizeI64 = fileSize;
                mediator.writeBytes(reinterpret_cast<const uint8_t*>(&fileSizeI64), sizeof(fileSizeI64));

                std::vector<char> buffer(fileSize);
                file.read(buffer.data(), fileSize);
                mediator.writeBytes(reinterpret_cast<const uint8_t*>(buffer.data()), fileSize);
                file.close();
                
                std::cerr << "[Puck]   - " << filename << " (" << fileSize << " bytes)" << std::endl;
            }

            mediator.flush();

            jniUtil->ReleaseIntArrayElements(env, idsJ, ids, JNI_ABORT);
            ::system(("rm -rf " + tempDir).c_str());
            delete inputVectors;
            jniUtil->DeleteLocalRef(env, parametersJ);

            std::cerr << "[Puck] Index creation complete!" << std::endl;

        } catch (...) {
            ::system(("rm -rf " + tempDir).c_str());
            throw;
        }

    } catch (const std::exception& e) {
        std::cerr << "[Puck ERROR] CreateIndexFromTemplate failed: " << e.what() << std::endl;
        throw std::runtime_error("CreateIndexFromTemplate failed: " + std::string(e.what()));
    }
}


// LoadIndex
jlong LoadIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject input) {
    if (input == nullptr) {
        throw std::runtime_error("IndexInput cannot be null");
    }

    try {
        stream::NativeEngineIndexInputMediator mediator(jniUtil, env, input);
        puck_stream::PuckOpenSearchIOReader reader(&mediator);

        uint32_t magic, version, dimension, numVectors;
        mediator.copyBytes(sizeof(magic), reinterpret_cast<uint8_t*>(&magic));
        if (magic != 0x5055434B) {
            throw std::runtime_error("Invalid Puck index format (bad magic)");
        }

        mediator.copyBytes(sizeof(version), reinterpret_cast<uint8_t*>(&version));
        mediator.copyBytes(sizeof(dimension), reinterpret_cast<uint8_t*>(&dimension));
        mediator.copyBytes(sizeof(numVectors), reinterpret_cast<uint8_t*>(&numVectors));

        std::cerr << "[Puck] Loading index: " << numVectors << " vectors, dim=" << dimension << std::endl;

        std::vector<int> ids(numVectors);
        mediator.copyBytes(numVectors * sizeof(int), reinterpret_cast<uint8_t*>(ids.data()));

        std::vector<float> vectors(numVectors * dimension);
        mediator.copyBytes(numVectors * dimension * sizeof(float), 
                         reinterpret_cast<uint8_t*>(vectors.data()));

        std::vector<uint32_t> cellAssignments(numVectors);
        mediator.copyBytes(numVectors * sizeof(uint32_t), 
                         reinterpret_cast<uint8_t*>(cellAssignments.data()));

        int32_t numAllFiles;
        mediator.copyBytes(sizeof(numAllFiles), reinterpret_cast<uint8_t*>(&numAllFiles));

        std::string tempDir = createUniqueTempDir("puck_load");

        try {
            std::cerr << "[Puck Load] Using temp directory: " << tempDir << std::endl;
            std::cerr << "[Puck] Extracting " << numAllFiles << " files..." << std::endl;
            
            for (int32_t i = 0; i < numAllFiles; i++) {
                int32_t nameLen;
                mediator.copyBytes(sizeof(nameLen), reinterpret_cast<uint8_t*>(&nameLen));

                std::vector<char> nameBuffer(nameLen + 1, '\0');
                mediator.copyBytes(nameLen, reinterpret_cast<uint8_t*>(nameBuffer.data()));
                std::string filename(nameBuffer.data(), nameLen);

                int64_t fileSize;
                mediator.copyBytes(sizeof(fileSize), reinterpret_cast<uint8_t*>(&fileSize));

                std::vector<char> fileBuffer(fileSize);
                mediator.copyBytes(fileSize, reinterpret_cast<uint8_t*>(fileBuffer.data()));

                std::string filepath = tempDir + "/" + filename;
                std::ofstream outFile(filepath, std::ios::binary);
                outFile.write(fileBuffer.data(), fileSize);
                outFile.close();
                
                std::cerr << "[Puck]   - " << filename << " (" << fileSize << " bytes)" << std::endl;
            }

            // ✅ Verify cell_assign.dat exists with correct size
            std::string cellAssignPath = tempDir + "/cell_assign.dat";
            struct stat cellStat;
            if (stat(cellAssignPath.c_str(), &cellStat) != 0) {
                throw std::runtime_error("cell_assign.dat not found at: " + cellAssignPath);
            }
            size_t expectedCellSize = numVectors * sizeof(uint32_t);
            if (cellStat.st_size != expectedCellSize) {
                std::cerr << "[Puck ERROR] cell_assign.dat size: " << cellStat.st_size 
                         << ", expected: " << expectedCellSize << std::endl;
                throw std::runtime_error("cell_assign.dat size mismatch");
            }
            std::cerr << "[Puck] Verified cell_assign.dat: " << cellStat.st_size << " bytes" << std::endl;

            // Read config
            std::string indexDatPath = tempDir + "/index.dat";
            puck::IndexConf conf = readIndexConf(indexDatPath);
            conf.total_point_count = numVectors;
            conf.feature_dim = dimension;

            std::cerr << "[Puck] Config: coarse=" << conf.coarse_cluster_count 
                     << ", fine=" << conf.fine_cluster_count 
                     << ", filter_nsq=" << conf.filter_nsq 
                     << ", nsq=" << conf.nsq << std::endl;

            // ✅ CRITICAL: Set gflags BEFORE creating PuckIndex
            google::SetCommandLineOption("index_path", tempDir.c_str());
            google::SetCommandLineOption("feature_dim", std::to_string(dimension).c_str());
            google::SetCommandLineOption("total_point_count", std::to_string(numVectors).c_str());
            
            // Set all filename flags
            google::SetCommandLineOption("cell_assign_file_name", "cell_assign.dat");
            google::SetCommandLineOption("filter_data_file_name", "filter_data.dat");
            google::SetCommandLineOption("pq_data_file_name", "pq_data.dat");
            google::SetCommandLineOption("coarse_codebook_file_name", "coarse_codebook.dat");
            google::SetCommandLineOption("fine_codebook_file_name", "fine_codebook.dat");
            google::SetCommandLineOption("filter_codebook_file_name", "filter_codebook.dat");
            google::SetCommandLineOption("pq_codebook_file_name", "pq_codebook.dat");
            google::SetCommandLineOption("index_file_name", "index.dat");
            
            // Set config parameters
            google::SetCommandLineOption("coarse_cluster_count", std::to_string(conf.coarse_cluster_count).c_str());
            google::SetCommandLineOption("fine_cluster_count", std::to_string(conf.fine_cluster_count).c_str());
            google::SetCommandLineOption("whether_filter", conf.whether_filter ? "true" : "false");
            google::SetCommandLineOption("filter_nsq", std::to_string(conf.filter_nsq).c_str());
            google::SetCommandLineOption("whether_pq", conf.whether_pq ? "true" : "false");
            google::SetCommandLineOption("nsq", std::to_string(conf.nsq).c_str());
            google::SetCommandLineOption("ks", std::to_string(conf.ks).c_str());

            // ✅ Verify gflag was set
            std::string verifyPath;
            google::GetCommandLineOption("index_path", &verifyPath);
            std::cerr << "[Puck] Verified index_path gflag: " << verifyPath << std::endl;
            
            if (verifyPath != tempDir) {
                throw std::runtime_error("gflag index_path mismatch: set=" + tempDir + ", got=" + verifyPath);
            }

            std::cerr << "[Puck] Creating PuckIndex..." << std::endl;
            auto index = std::make_unique<PuckIndexWrapper>();
            index->public_set_conf(conf);
            
            std::cerr << "[Puck] Initializing model memory..." << std::endl;
            if (index->public_init_model_memory() != 0) {
                throw std::runtime_error("Failed to initialize model memory");
            }
            
            std::cerr << "[Puck] Reading codebooks from: " << tempDir << std::endl;
            if (index->public_read_coodbooks() != 0) {
                throw std::runtime_error("Failed to read codebooks");
            }

            std::cerr << "[Puck] Building memory mappings (will read from " << tempDir << ")..." << std::endl;
            std::unique_ptr<uint32_t[]> cellStartMemoryIdx(
                new uint32_t[conf.coarse_cluster_count * conf.fine_cluster_count + 1]
            );
            std::unique_ptr<uint32_t[]> localToMemoryIdx(new uint32_t[numVectors]);
            
            int convertResult = index->public_convert_local_to_memory_idx(
                cellStartMemoryIdx.get(), 
                localToMemoryIdx.get()
            );
            
            if (convertResult != 0) {
                std::cerr << "[Puck ERROR] convert_local_to_memory_idx failed with code: " << convertResult << std::endl;
                std::cerr << "[Puck ERROR] Current gflag index_path: " << verifyPath << std::endl;
                std::cerr << "[Puck ERROR] Expected path: " << tempDir << std::endl;
                throw std::runtime_error("Failed to convert indices");
            }
            
            std::cerr << "[Puck] Loading quantization features..." << std::endl;
            int readResult = index->public_read_feature_index(localToMemoryIdx.get());
            
            if (readResult != 0) {
                throw std::runtime_error("Failed to read feature index");
            }

            auto* wrapper = new IndexMetadata();
            wrapper->ids = std::move(ids);
            wrapper->vectors = std::move(vectors);
            wrapper->cellAssignments = std::move(cellAssignments);
            wrapper->puckIndex = std::move(index);
            wrapper->indexConf = conf;
            wrapper->dimension = dimension;
            wrapper->numVectors = numVectors;

            ::system(("rm -rf " + tempDir).c_str());

            std::cerr << "[Puck] Index loaded successfully!" << std::endl;
            return reinterpret_cast<jlong>(wrapper);

        } catch (...) {
            ::system(("rm -rf " + tempDir).c_str());
            throw;
        }

    } catch (const std::exception& e) {
        std::cerr << "[Puck ERROR] LoadIndex failed: " << e.what() << std::endl;
        throw std::runtime_error("LoadIndex failed: " + std::string(e.what()));
    }
}





// Query
int knn_query_index(jlong indexPointer, jfloat* queryVector, jint dimension, jint k,
                    const std::unordered_map<std::string, jobject>& parameters,
                    jfloat* distances, jlong* indices) {
    
    if (indexPointer == 0) {
        throw std::invalid_argument("Index pointer is null");
    }

    try {
        auto* metadata = reinterpret_cast<IndexMetadata*>(indexPointer);

        puck::SearchContext searchContext;
        puck::IndexConf& conf = metadata->indexConf;
        
        if (conf.topk < static_cast<uint32_t>(k)) {
            conf.topk = k;
        }

        int resetResult = searchContext.reset(conf);
        if (resetResult != 0) {
            throw std::runtime_error("Failed to initialize SearchContext");
        }

        puck::Request request;
        request.feature = queryVector;
        request.topk = k;

        puck::Response response;
        std::vector<float> distancesVec(k);
        std::vector<uint32_t> memoryIndices(k);
        response.distance = distancesVec.data();
        response.local_idx = memoryIndices.data();
        response.result_num = 0;

        int searchResult = metadata->puckIndex->public_search(&request, &response);

        if (searchResult != 0) {
            throw std::runtime_error("Search failed: " + std::to_string(searchResult));
        }

        int resultCount = std::min(k, static_cast<int>(response.result_num));
        
        for (int i = 0; i < resultCount; i++) {
            uint32_t memIdx = response.local_idx[i];
            
            if (memIdx >= metadata->ids.size()) {
                throw std::runtime_error("Invalid memory index");
            }

            distances[i] = response.distance[i];
            indices[i] = static_cast<jlong>(metadata->ids[memIdx]);
        }

        return resultCount;

    } catch (const std::exception& e) {
        std::cerr << "[Puck Query ERROR] " << e.what() << std::endl;
        throw;
    }
}

// Free index
void knn_free_index(jlong indexPointer) {
    if (indexPointer != 0) {
        auto* metadata = reinterpret_cast<IndexMetadata*>(indexPointer);
        delete metadata;
    }
}

// Stub implementations
jlong knn_new_index(jlong numDocs, jint dimension, const std::unordered_map<std::string, jobject>& parameters) {
    throw std::runtime_error("Use trainIndex + createIndexFromTemplate instead");
}

void knn_train_index(jlong indexPointer, jfloat* trainingVectors, jint numVectors, jint dimension,
                     const std::unordered_map<std::string, jobject>& parameters) {
    throw std::runtime_error("Use trainIndex + createIndexFromTemplate instead");
}

void CreateIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray idsJ,
                 jlong vectorsAddressJ, jint dimJ, jobject output, jobject parametersJ) {
    throw std::runtime_error("Use trainIndex + createIndexFromTemplate instead");
}

} // namespace puck_wrapper
} // namespace knn_jni
