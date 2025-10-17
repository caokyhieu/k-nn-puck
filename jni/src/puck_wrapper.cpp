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
#include <glog/logging.h>

jint JNI_OnLoad(JavaVM* vm, void*) {
    google::InitGoogleLogging("opensearchknn_puck");
    return JNI_VERSION_1_8;
}

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

    void directSetConf(const puck::IndexConf &conf, const std::string &tempDir) {
        _conf = conf;
        _conf.index_path = tempDir;
        _conf.coarse_codebook_file_name = tempDir + "/coarse_codebook.dat";
        _conf.fine_codebook_file_name   = tempDir + "/fine_codebook.dat";
        _conf.filter_codebook_file_name = tempDir + "/filter_codebook.dat";
        _conf.filter_data_file_name     = tempDir + "/filter_data.dat";
        _conf.pq_codebook_file_name     = tempDir + "/pq_codebook.dat";
        _conf.pq_data_file_name         = tempDir + "/pq_data.dat";
        _conf.cell_assign_file_name     = tempDir + "/cell_assign.dat";
        _conf.index_file_name           = tempDir + "/index.dat";
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

#define PUCK_LOG(msg) std::cout << "[PUCK JNI] " << msg << std::endl;

struct IndexMetadata {
    std::vector<int> ids;
    std::vector<float> vectors;
    std::vector<uint32_t> cellAssignments;
    uint32_t dimension;
    uint32_t numVectors;
    std::unique_ptr<PuckIndexWrapper> puckIndex;
    puck::IndexConf indexConf;
    std::string tempDir;  // Temp dir to clean up

    ~IndexMetadata() {
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


// Full corrected LoadIndex
jlong LoadIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject input) {
    if (input == nullptr) {
        jniUtil->ThrowJavaException(env, "java/lang/NullPointerException", "IndexInput is null");
        return 0;
    }

    std::string tempDir;
    try {
        stream::NativeEngineIndexInputMediator mediator(jniUtil, env, input);

        // Read header
        uint32_t magic = 0;
        mediator.copyBytes(sizeof(magic), reinterpret_cast<uint8_t*>(&magic));
        const uint32_t EXPECTED_MAGIC = 0x5055434B;
        if (magic != EXPECTED_MAGIC) {
            throw std::runtime_error("Invalid Puck index format (bad magic)");
        }

        uint32_t version;
        mediator.copyBytes(sizeof(version), reinterpret_cast<uint8_t*>(&version));
        // Optionally validate version

        uint32_t dimension, numVectors;
        mediator.copyBytes(sizeof(dimension), reinterpret_cast<uint8_t*>(&dimension));
        mediator.copyBytes(sizeof(numVectors), reinterpret_cast<uint8_t*>(&numVectors));

        LOG(INFO) << "[Puck] Loading index: " << numVectors << " vectors, dim=" << dimension;

        // Read IDs, raw vectors, cell assignments
        std::vector<int> ids(numVectors);
        mediator.copyBytes(numVectors * sizeof(int),
                           reinterpret_cast<uint8_t*>(ids.data()));

        std::vector<float> vectors(numVectors * dimension);
        mediator.copyBytes(numVectors * dimension * sizeof(float),
                           reinterpret_cast<uint8_t*>(vectors.data()));

        std::vector<uint32_t> cellAssignments(numVectors);
        mediator.copyBytes(numVectors * sizeof(uint32_t),
                           reinterpret_cast<uint8_t*>(cellAssignments.data()));

        // Read number of files
        int32_t numAllFiles = 0;
        mediator.copyBytes(sizeof(int32_t),
                           reinterpret_cast<uint8_t*>(&numAllFiles));
        if (numAllFiles <= 0) {
            throw std::runtime_error("Invalid file count in serialized index");
        }

        // Create a temp directory for extraction
        tempDir = createUniqueTempDir("puck_load");

        // Extract model files
        std::vector<std::string> filenames;
        filenames.reserve(numAllFiles);
        for (int i = 0; i < numAllFiles; i++) {
            int32_t nameLen = 0;
            mediator.copyBytes(sizeof(nameLen),
                               reinterpret_cast<uint8_t*>(&nameLen));
            if (nameLen <= 0 || nameLen > 1024) {
                throw std::runtime_error("Invalid filename length " + std::to_string(nameLen));
            }
            std::vector<char> nameBuf(nameLen);
            mediator.copyBytes(nameLen,
                               reinterpret_cast<uint8_t*>(nameBuf.data()));
            std::string fname(nameBuf.begin(), nameBuf.end());

            int64_t fileSize = 0;
            mediator.copyBytes(sizeof(fileSize),
                               reinterpret_cast<uint8_t*>(&fileSize));
            if (fileSize < 0) {
                throw std::runtime_error("Invalid file size for " + fname);
            }

            std::vector<char> fileBuf(fileSize);
            if (fileSize > 0) {
                mediator.copyBytes(fileSize,
                                   reinterpret_cast<uint8_t*>(fileBuf.data()));
            }

            std::string filepath = tempDir + "/" + fname;
            std::ofstream of(filepath, std::ios::binary);
            of.write(fileBuf.data(), fileBuf.size());
            of.close();

            filenames.push_back(fname);
            LOG(INFO) << "[Puck]   - Extracted " << fname << " (" << fileSize << " bytes)";
        }

        // Now we must read config (index.dat) so we know filter_nsq, nsq, cluster counts, etc.
        std::string indexDatPath = tempDir + "/index.dat";
        puck::IndexConf conf = readIndexConf(indexDatPath);
        // Override conf.number_of_vectors from serialized data
        conf.feature_dim = dimension;
        conf.total_point_count = numVectors;

        LOG(INFO) << "[Puck] Config loaded: coarse=" << conf.coarse_cluster_count
                  << " fine=" << conf.fine_cluster_count
                  << " filter_nsq=" << conf.filter_nsq
                  << " nsq=" << conf.nsq
                  << " whether_filter=" << conf.whether_filter
                  << " whether_pq=" << conf.whether_pq;

        // IMPORTANT: set all gflags *before* creating/initializing the PuckIndex
        google::SetCommandLineOption("index_path", tempDir.c_str());
        google::SetCommandLineOption("feature_dim", std::to_string(dimension).c_str());
        google::SetCommandLineOption("total_point_count", std::to_string(numVectors).c_str());
        google::SetCommandLineOption("coarse_cluster_count", std::to_string(conf.coarse_cluster_count).c_str());
        google::SetCommandLineOption("fine_cluster_count", std::to_string(conf.fine_cluster_count).c_str());
        google::SetCommandLineOption("whether_filter", (conf.whether_filter ? "true" : "false"));
        google::SetCommandLineOption("filter_nsq", std::to_string(conf.filter_nsq).c_str());
        google::SetCommandLineOption("whether_pq", (conf.whether_pq ? "true" : "false"));
        google::SetCommandLineOption("nsq", std::to_string(conf.nsq).c_str());
        google::SetCommandLineOption("ks", std::to_string(conf.ks).c_str());
        // Also set file names flags if needed (optional, if consistent with build)
        // e.g. coarse_codebook_file_name, etc.

        // Sanity-check: ensure that index_path gflag was set correctly
        std::string verifyIndexPath;
        google::GetCommandLineOption("index_path", &verifyIndexPath);
        if (verifyIndexPath != tempDir) {
            throw std::runtime_error("gflag index_path mismatch: set " + tempDir + " got " + verifyIndexPath);
        }

        // Instantiate PuckIndex and load into memory
        auto index = std::make_unique<PuckIndexWrapper>();
        index->directSetConf(conf, tempDir);

   
        // Initialize model memory
        if (index->public_init_model_memory() != 0) {
            throw std::runtime_error("init_model_memory failed");
        }

        // Load codebooks / quantization tables
        if (index->public_read_coodbooks() != 0) {
            throw std::runtime_error("read_codebooks failed");
        }

        // Set up memory indexing
        auto coarseCount = conf.coarse_cluster_count * conf.fine_cluster_count;
        std::unique_ptr<uint32_t[]> cellStartMemoryIdx(new uint32_t[coarseCount + 1]);
        std::unique_ptr<uint32_t[]> localToMemoryIdx(new uint32_t[numVectors]);

        // int conv = index->public_convert_local_to_memory_idx(cellStartMemoryIdx.get(), localToMemoryIdx.get());
        // if (conv != 0) {
        //     throw std::runtime_error("convert_local_to_memory_idx failed (code " + std::to_string(conv) + ")");
        // }
        // auto coarseCount = conf.coarse_cluster_count * conf.fine_cluster_count;
        // std::unique_ptr<uint32_t[]> cell_start_memory_idx(new uint32_t[coarseCount + 1]);
        // std::unique_ptr<uint32_t[]> local_to_memory_idx(new uint32_t[numVectors]);
        for (uint32_t i = 0; i < numVectors; i++) {
            localToMemoryIdx[i] = i;  // Identity mapping
        }

        // Build cell_start_memory_idx from cellAssignments
        std::vector<uint32_t> cell_counts(coarseCount, 0);
        for (uint32_t i = 0; i < numVectors; i++) {
            cell_counts[cellAssignments[i]]++;
        }

        cellStartMemoryIdx[0] = 0;
        for (uint32_t i = 0; i < coarseCount; i++) {
            cellStartMemoryIdx[i + 1] = cellStartMemoryIdx[i] + cell_counts[i];
        }
        
        // Load quantization features (PQ / filter data)
        if (index->public_read_feature_index(localToMemoryIdx.get()) != 0) {
            throw std::runtime_error("read_feature_index failed");
        }

        // Wrap into metadata to return to Java
        IndexMetadata *meta = new IndexMetadata();
        meta->ids = std::move(ids);
        meta->vectors = std::move(vectors);
        meta->cellAssignments = std::move(cellAssignments);
        meta->puckIndex = std::move(index);
        meta->indexConf = conf;
        meta->dimension = dimension;
        meta->numVectors = numVectors;
        meta->tempDir = tempDir;

        LOG(INFO) << "[Puck] LoadIndex succeeded, returning native pointer";
        return reinterpret_cast<jlong>(meta);

    } catch (const std::exception &e) {
        // Clean up tempDir if created
        if (!tempDir.empty()) {
            ::system(("rm -rf " + tempDir).c_str());
        }
        std::string msg = std::string("LoadIndex failed: ") + e.what();
        jniUtil->ThrowJavaException(env, "java/lang/Exception", msg.c_str());
        return 0;
    }
}

jobjectArray QueryIndex(
    JNIUtilInterface* jniUtil,
    JNIEnv* env,
    jlong indexPointerJ,
    jfloatArray queryVectorJ,
    jint kJ,
    jobject methodParamsJ,
    jlongArray filterIdsJ,      // Can skip initially
    jint filterIdsTypeJ     // ← Include but don't implement yet
){
    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }
    
    auto* metadata = reinterpret_cast<IndexMetadata*>(indexPointerJ);
    if (!metadata || !metadata->puckIndex) {
        throw std::runtime_error("Invalid index pointer");
    }
    
    // Extract query vector
    float* rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
    int dimension = jniUtil->GetJavaFloatArrayLength(env, queryVectorJ);
    
    // Prepare buffers
    std::vector<float> distances(kJ);
    std::vector<uint32_t> localIndices(kJ);
    


    
    try {
        // Normalize query (CRITICAL!)
        std::vector<float> normalizedQuery(dimension);
        float norm = cblas_snrm2(dimension, rawQueryVector, 1);
        
        if (norm > 1e-6) {
            float invNorm = 1.0f / norm;
            for (int i = 0; i < dimension; i++) {
                normalizedQuery[i] = rawQueryVector[i] * invNorm;
            }
        } else {
            std::copy(rawQueryVector, rawQueryVector + dimension, normalizedQuery.begin());
        }
        
        // Prepare request
        puck::Request request;
        request.feature = normalizedQuery.data();
        request.topk = kJ;
        
        puck::Response response;
        response.distance = distances.data();
        response.local_idx = localIndices.data();
        response.result_num = 0;
        
        // Use search() not public_search()
        int result = metadata->puckIndex->search(&request, &response);
        
        if (result != 0) {
            throw std::runtime_error("Search failed: " + std::to_string(result));
        }
        
        int resultSize = std::min(kJ, static_cast<int>(response.result_num));
        
        // Create Java result array (FAISS style)
        jclass resultClass = jniUtil->FindClass(env, "org/opensearch/knn/index/query/KNNQueryResult");
        jmethodID constructor = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");
        jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);
        
        for (int i = 0; i < resultSize; i++) {
            uint32_t localIdx = response.local_idx[i];
            if (localIdx >= metadata->ids.size()) {
                throw std::runtime_error("Invalid local index");
            }
            
            jlong docId = static_cast<jlong>(metadata->ids[localIdx]);
            jfloat dist = distances[i];
            
            jobject resultObj = jniUtil->NewObject(env, resultClass, constructor, docId, dist);
            jniUtil->SetObjectArrayElement(env, results, i, resultObj);
        }
        
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
        return results;
        
    } catch (...) {
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
        throw;
    }
}


// Query
int knn_query_index(jlong indexPointer, jfloat* queryVector, jint dimension, 
                    jint k, const std::unordered_map<std::string, jobject>& parameters,
                    jfloat* distances, jlong* indices) {
    
    LOG(INFO) << "=== PUCK QUERY START ===";
    LOG(INFO) << "k=" << k << ", dim=" << dimension;
    
    if (indexPointer == 0) {
        LOG(ERROR) << "Null index pointer";
        return 0;
    }

    PUCK_LOG("---- KNN Query Start ----");
    PUCK_LOG("Index pointer: " << indexPointer);
    PUCK_LOG("Query dimension: " << dimension);
    PUCK_LOG("k (neighbors): " << k);

    try {
        auto* metadata = reinterpret_cast<IndexMetadata*>(indexPointer);
        
        if (!metadata->puckIndex) {
            LOG(ERROR) << "PuckIndex is null";
            return 0;
        }
        
        LOG(INFO) << "Index has " << metadata->ids.size() << " documents";
        
        // Get configuration
        puck::IndexConf& conf = metadata->puckIndex->get_index_conf();
        
        // Ensure topk is sufficient
        if (conf.topk < static_cast<uint32_t>(k)) {
            conf.topk = k;
        }
        
        // Normalize query vector (critical!)
        std::vector<float> normalizedQuery(dimension);
        float norm = 0.0f;
        for (int i = 0; i < dimension; i++) {
            norm += queryVector[i] * queryVector[i];
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-6) {
            for (int i = 0; i < dimension; i++) {
                normalizedQuery[i] = queryVector[i] / norm;
            }
            LOG(INFO) << "Query normalized (norm=" << norm << ")";
        } else {
            LOG(WARNING) << "Zero norm query, copying as-is";
            std::copy(queryVector, queryVector + dimension, normalizedQuery.begin());
        }
        
        // Prepare request
        puck::Request request;
        request.feature = normalizedQuery.data();
        request.topk = k;
        
        // Prepare response buffers
        std::vector<float> distancesVec(k, std::numeric_limits<float>::max());
        std::vector<uint32_t> localIndices(k, 0);
        
        puck::Response response;
        response.distance = distancesVec.data();
        response.local_idx = localIndices.data();
        response.result_num = 0;
        
        LOG(INFO) << "Calling PuckIndex::search()";
        
        // Use the main search() method - it manages context pool internally
        int searchResult = metadata->puckIndex->search(&request, &response);
        
        LOG(INFO) << "Search result: " << searchResult 
                  << ", found: " << response.result_num;
        
        if (searchResult != 0) {
            LOG(ERROR) << "Search failed: " << searchResult;
            return 0;
        }
        
        // Copy results
        int resultCount = std::min(k, static_cast<int>(response.result_num));
        
        for (int i = 0; i < resultCount; i++) {
            uint32_t localIdx = response.local_idx[i];
            
            if (localIdx >= metadata->ids.size()) {
                LOG(ERROR) << "Invalid local index: " << localIdx;
                return 0;
            }
            
            distances[i] = response.distance[i];
            indices[i] = static_cast<jlong>(metadata->ids[localIdx]);
            
            LOG(INFO) << "  [" << i << "] doc=" << indices[i] 
                      << " dist=" << distances[i];
        }
        
        LOG(INFO) << "=== PUCK QUERY END: " << resultCount << " results ===";
        return resultCount;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Query exception: " << e.what();
        return 0;
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
