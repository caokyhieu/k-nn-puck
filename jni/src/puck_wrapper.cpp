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
#include <filesystem>

jint JNI_OnLoad(JavaVM* vm, void*) {
    google::InitGoogleLogging("opensearchknn_puck");
    // Configure glog to output to stderr (so it appears in OpenSearch logs)
    FLAGS_logtostderr = true;           // Send logs to stderr
    FLAGS_stderrthreshold = google::INFO; // Log INFO and above to stderr
    FLAGS_minloglevel = google::INFO;   // Minimum log level
    FLAGS_colorlogtostderr = false;     // No color codes
    FLAGS_logbufsecs = 0;               // Flush immediately
    
    std::cerr << "[Puck-JNI] Library loaded, glog configured to stderr" << std::endl;
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


namespace knn_jni {
namespace puck_wrapper {

#define PUCK_LOG(msg) std::cout << "[PUCK JNI] " << msg << std::endl;



struct IndexMetadata {
    std::vector<int> ids;
    std::vector<float> vectors;
    std::vector<uint32_t> cellAssignments;
    std::unique_ptr<puck::PuckIndex> puckIndex;  // Can point to any index type
    puck::IndexConf indexConf;
    std::string tempDir;  // Temp dir to clean up
    uint32_t dimension;
    uint32_t numVectors;

   
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


        // write feature file to index later , need to clean that up after indexing
        try {
            std::string vectorsFile = tempDir + "/" + "all_data.feat.bin"; // hard code file name
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

            // init PuckIndex
            puck::PuckIndex* puckIdx = new puck::PuckIndex();
            
            if (!puckIdx) {
                throw std::runtime_error("Failed to create PuckIndex");
            }

            std::cerr << "[Puck] Starting training..." << std::endl;
            int trainResult = puckIdx->train();
            if (trainResult != 0) {
                throw std::runtime_error("Training failed: " + std::to_string(trainResult));
            }
            std::cerr << "[Puck] Training completed" << std::endl;


            // Serialize model files, take from puckIdx->_conf, these are 5 important fields decide to the index result
           std::vector<std::string> modelFiles = {
                "coarse.dat",
                "fine.dat",
                "filter_codebook.dat", 
                "learn_codebooks.dat",
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

            jniUtil->DeleteLocalRef(env, parametersJ);

            std::cerr << "[Puck] Training complete, model size=" << serialized.size() << " bytes" << std::endl;

            // delete temp dir
            std::filesystem::remove_all(tempDir);
            std::cerr << "[Puck] Deleted temporary directory: " << tempDir << std::endl;
            return result;

        } catch (...) {
            throw;
        }

    } catch (const std::exception& e) {
        jniUtil->ThrowJavaException(env, "java/lang/Exception", e.what());
        return nullptr;
    }
}


jlong LoadIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jstring indexPathJ) {
    if (indexPathJ == nullptr) {
        jniUtil->ThrowJavaException(env, "java/lang/NullPointerException", "Index path is null");
        return 0;
    }

    try {
        
        std::string indexPath(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
        
        //set index type and path
        google::SetCommandLineOption("index_path", indexPath.c_str());

        puck::PuckIndex* index = new puck::PuckIndex();
        index->init();

        if (!index) {
            throw std::runtime_error("Failed to create PuckIndex");
        }
        std::cerr << "[Puck] Loading index from: " << indexPath << std::endl;
        //set the path
       
        // read dimension and num vectors 
        // Create metadata
        IndexMetadata* meta = new IndexMetadata();
        meta->puckIndex.reset(index);  // Transfer ownership to unique_ptr
        meta->indexConf = readIndexConf(indexPath + '/' + "index.dat");
        meta->dimension = meta->indexConf.feature_dim;
        meta->numVectors = meta->indexConf.total_point_count;
        meta->tempDir = indexPath;
        // need more
        
        // Note: You need to decide how to store/retrieve IDs and raw vectors for file-based loading

        return reinterpret_cast<jlong>(meta);

    } catch (const std::exception& e) {
        jniUtil->ThrowJavaException(env, "java/lang/Exception", e.what());
        return 0;
    }
}



void CreateIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jintArray idsJ,
                             jlong vectorsAddressJ, jint dimJ, jobject output,
                             jbyteArray templateIndexJ, jobject parametersJ) {
    if (idsJ == nullptr || vectorsAddressJ <= 0 || dimJ <= 0 || 
        output == nullptr || templateIndexJ == nullptr) {
        throw std::runtime_error("Invalid parameters");
    }

   
    // read vectors to build index 
    try {
        auto* inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
        int dim = static_cast<int>(dimJ);
        int numVectors = inputVectors->size() / dim;
        int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);

        if (numIds != numVectors) {
            throw std::runtime_error("ID count mismatch");
        }

        std::cerr << "[Puck] Creating index: " << numVectors << " vectors, dim=" << dim << std::endl;

        // parse the important files for PuckIndex creation from serialized templateIndexJ (5 files when we trained index)
        int templateSize = jniUtil->GetJavaBytesArrayLength(env, templateIndexJ);
        jbyte* templateBytes = jniUtil->GetByteArrayElements(env, templateIndexJ, nullptr);

        // create temp dir to save index files for building
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

                  // ✅ Set gflags BEFORE init_single_build
            google::SetCommandLineOption("index_path", tempDir.c_str()); // just need to set the index folder(temp dir) we created, puck will build from index files

            //  create index
            puck::Index* index = new puck::PuckIndex();

            // ✅ Read config FIRST
            std::string indexDatPath = tempDir + '/' + "index.dat"; // this file just saved from previous step
            puck::IndexConf conf = readIndexConf(indexDatPath);

            std::cerr << "[Puck] Config: filter_nsq=" << conf.filter_nsq 
                     << ", nsq=" << conf.nsq << ", ks=" << conf.ks << std::endl;

            // ✅ Calculate sizes from config
            const int filterBytesPerVector = sizeof(float) + conf.filter_nsq;
            const int pqBytesPerVector = sizeof(float) + conf.nsq;

            std::cerr << "[Puck] Quantization format: filter=" << filterBytesPerVector 
                     << " bytes/vec, pq=" << pqBytesPerVector << " bytes/vec" << std::endl;

      
            
            std::cerr << "[Puck] Initializing PuckIndex for building..." << std::endl;
            

            jint* ids = jniUtil->GetIntArrayElements(env, idsJ, nullptr);
            std::cerr << "[Puck] Building index with " << numVectors << " vectors..." << std::endl;

            
            // we write the vectors in fvecs format into folder tempDir
            // using function like in TrainIndex
            try {
            std::string vectorsFile = tempDir + '/' + "all_data.feat.bin";
            std::ofstream out(vectorsFile, std::ios::binary);
            if (!out.is_open()) {
                throw std::runtime_error("Failed to create training file");
            }

            for (int i = 0; i < numVectors; i++) {
                out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
                const float* vec = inputVectors->data() + (i * dim);
                out.write(reinterpret_cast<const char*>(vec), dim * sizeof(float));
            }
            out.close();
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to write vectors for building: " + std::string(e.what()));
            }

            // init for build, just after writing vectors features
            index->init(); // remember to init after settting the directory

            // after writing vectors, we can build the index
            if (index->build() != 0) {
                throw std::runtime_error("Build failed");
            }
            std::cerr << "[Puck] Index build completed" << std::endl;
            // after building, puck will generate cell_assign.dat file
            // Read cell assignments
            std::string cellAssignPath = tempDir + '/' + "cell_assign.dat";
            std::ifstream cellFile(cellAssignPath, std::ios::binary);
            std::vector<uint32_t> cellAssignments(numVectors);
            cellFile.read(reinterpret_cast<char*>(cellAssignments.data()),
                        numVectors * sizeof(uint32_t));
            cellFile.close();

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
            // after building, we have extra 3 files to add: filter_data.dat, pq_data.dat, cell_assign.dat
            allFiles.push_back("filter_data.dat");
            allFiles.push_back("pq_data.dat");
            allFiles.push_back("cell_assign.dat");


            int32_t numAllFiles = allFiles.size();
            mediator.writeBytes(reinterpret_cast<const uint8_t*>(&numAllFiles), sizeof(numAllFiles));

            std::cerr << "[Puck] Writing " << numAllFiles << " files to index..." << std::endl;
            // serialize each file
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
            delete inputVectors;
            jniUtil->DeleteLocalRef(env, parametersJ);

            std::cerr << "[Puck] Index creation complete!" << std::endl;
            // delete temp dir
            std::filesystem::remove_all(tempDir);
            std::cerr << "[Puck] Deleted temporary directory: " << tempDir << std::endl;

        } catch (...) {
            throw;
        }

    } catch (const std::exception& e) {
        std::cerr << "[Puck ERROR] CreateIndexFromTemplate failed: " << e.what() << std::endl;
        throw std::runtime_error("CreateIndexFromTemplate failed: " + std::string(e.what()));
    }
}



// Full corrected LoadIndex
jlong LoadIndexWithStream(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jobject input) {
    std::cerr << "[Puck] LoadIndexWithStream called" << std::endl;  // ← Use cerr
    
    if (input == nullptr) {
        std::cerr << "[Puck] ERROR: IndexInput is null" << std::endl;
        jniUtil->ThrowJavaException(env, "java/lang/NullPointerException", "IndexInput is null");
        return 0;
    }

    std::string tempDir;
    try {
        // read from input stream serialized index
        stream::NativeEngineIndexInputMediator mediator(jniUtil, env, input);
        std::cerr << "[Puck] Created mediator" << std::endl;

        // Read header
        uint32_t magic = 0;
        mediator.copyBytes(sizeof(magic), reinterpret_cast<uint8_t*>(&magic));
        std::cerr << "[Puck] Read magic: 0x" << std::hex << magic << std::dec << std::endl;
        
        const uint32_t EXPECTED_MAGIC = 0x5055434B;
        if (magic != EXPECTED_MAGIC) {
            std::cerr << "[Puck] ERROR: Invalid magic" << std::endl;
            throw std::runtime_error("Invalid Puck index format (bad magic)");
        }

        uint32_t version;
        mediator.copyBytes(sizeof(version), reinterpret_cast<uint8_t*>(&version));
        
        uint32_t dimension, numVectors;
        mediator.copyBytes(sizeof(dimension), reinterpret_cast<uint8_t*>(&dimension));
        mediator.copyBytes(sizeof(numVectors), reinterpret_cast<uint8_t*>(&numVectors));

        std::cerr << "[Puck] Loading index: " << numVectors << " vectors, dim=" << dimension << std::endl;

        // Read IDs, raw vectors, cell assignments
        std::vector<int> ids(numVectors);
        mediator.copyBytes(numVectors * sizeof(int),
                           reinterpret_cast<uint8_t*>(ids.data()));
        std::cerr << "[Puck] Read " << numVectors << " IDs" << std::endl;

        std::vector<float> vectors(numVectors * dimension);
        mediator.copyBytes(numVectors * dimension * sizeof(float),
                           reinterpret_cast<uint8_t*>(vectors.data()));
        std::cerr << "[Puck] Read " << numVectors << " vectors" << std::endl;

        std::vector<uint32_t> cellAssignments(numVectors);
        mediator.copyBytes(numVectors * sizeof(uint32_t),
                           reinterpret_cast<uint8_t*>(cellAssignments.data()));
        std::cerr << "[Puck] Read cell assignments" << std::endl;

        // Read number of files
        int32_t numAllFiles = 0;
        mediator.copyBytes(sizeof(int32_t),
                           reinterpret_cast<uint8_t*>(&numAllFiles));
        std::cerr << "[Puck] Number of files to extract: " << numAllFiles << std::endl;
        
        if (numAllFiles <= 0) {
            throw std::runtime_error("Invalid file count in serialized index");
        }

        // Create temp directory
        tempDir = createUniqueTempDir("puck_load");
        std::cerr << "[Puck] Created temp dir: " << tempDir << std::endl;

        // Extract model files and write to temp dir
        for (int i = 0; i < numAllFiles; i++) {
            int32_t nameLen = 0;
            mediator.copyBytes(sizeof(nameLen),
                               reinterpret_cast<uint8_t*>(&nameLen));
            
            std::vector<char> nameBuf(nameLen);
            mediator.copyBytes(nameLen,
                               reinterpret_cast<uint8_t*>(nameBuf.data()));
            std::string fname(nameBuf.begin(), nameBuf.end());

            int64_t fileSize = 0;
            mediator.copyBytes(sizeof(fileSize),
                               reinterpret_cast<uint8_t*>(&fileSize));

            std::vector<char> fileBuf(fileSize);
            if (fileSize > 0) {
                mediator.copyBytes(fileSize,
                                   reinterpret_cast<uint8_t*>(fileBuf.data()));
            }

            std::string filepath = tempDir + "/" + fname;
            std::ofstream of(filepath, std::ios::binary);
            of.write(fileBuf.data(), fileBuf.size());
            of.close();

            std::cerr << "[Puck]   Extracted " << fname << " (" << fileSize << " bytes)" << std::endl;
        }

        std::cerr << "[Puck] All files extracted" << std::endl;
        // Set temp dir to gflags for puck to load files
        google::SetCommandLineOption("index_path", tempDir.c_str());

        // Create and initialize index
        puck::PuckIndex* index = new puck::PuckIndex();
        
        // Read config
        std::string indexDatPath = tempDir + "/" + "index.dat";
        puck::IndexConf conf = readIndexConf(indexDatPath);
        conf.feature_dim = dimension;
        conf.total_point_count = numVectors;

        std::cerr << "[Puck] Config loaded: coarse=" << conf.coarse_cluster_count
                  << " fine=" << conf.fine_cluster_count
                  << " filter_nsq=" << conf.filter_nsq
                  << " nsq=" << conf.nsq << std::endl;

        
        index->init();

        
        std::cerr << "[Puck] PuckIndex initialized" << std::endl;

        // Create metadata
        IndexMetadata *meta = new IndexMetadata();
        meta->ids = std::move(ids);
        meta->vectors = std::move(vectors);
        meta->cellAssignments = std::move(cellAssignments);
        meta->puckIndex.reset(index);
        meta->indexConf = conf;
        meta->dimension = dimension;
        meta->numVectors = numVectors;
        meta->tempDir = tempDir;

        std::cerr << "[Puck] LoadIndex SUCCEEDED, returning pointer" << std::endl;
        return reinterpret_cast<jlong>(meta);

    } catch (const std::exception &e) {
        std::cerr << "[Puck] LoadIndex FAILED: " << e.what() << std::endl;
        jniUtil->ThrowJavaException(env, "java/lang/Exception", e.what());
        return 0;
    }
}



// jobjectArray QueryIndex(
//     knn_jni::JNIUtilInterface* jniUtil,
//     JNIEnv* env,
//     jlong indexPointerJ,
//     jfloatArray queryVectorJ,
//     jint kJ,
//     jobject methodParamsJ,
//     jintArray parentIdsJ
// ) {
//     std::cerr << "[Puck-JNI] ===== QueryIndex START =====" << std::endl;
//     std::cerr << "[Puck-JNI] indexPointer=" << indexPointerJ << " k=" << kJ << std::endl;
    
//     try {
//         // 1. Get metadata
//         auto* metadata = reinterpret_cast<IndexMetadata*>(indexPointerJ);
//         if (!metadata) {
//             std::cerr << "[Puck-JNI] ERROR: metadata is NULL" << std::endl;
//             throw std::runtime_error("Invalid metadata pointer");
//         }
        
//         if (!metadata->puckIndex) {
//             std::cerr << "[Puck-JNI] ERROR: puckIndex is NULL" << std::endl;
//             throw std::runtime_error("Invalid puckIndex pointer");
//         }

                
//         std::cerr << "[Puck-JNI] Index has " << metadata->ids.size() << " documents" << std::endl;
//         std::cerr << "[Puck-JNI] Segment info:" << std::endl;
//         std::cerr << "  - Document count: " << metadata->ids.size() << std::endl;
//         std::cerr << "  - Dimension: " << metadata->dimension << std::endl;
//         std::cerr << "  - Temp dir: " << metadata->tempDir << std::endl;
//         std::cerr << "  - First doc ID: " << (metadata->ids.empty() ? -1 : metadata->ids[0]) << std::endl;
//         std::cerr << "  - Last doc ID: " << (metadata->ids.empty() ? -1 : metadata->ids.back()) << std::endl;
//         std::cerr << "[Puck-JNI] PuckIndex pointer: " << metadata->puckIndex.get() << std::endl;
        
//         // 2. Extract query vector
//         float* queryArray = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
//         jsize dimension = jniUtil->GetJavaFloatArrayLength(env, queryVectorJ);
        
//         std::cerr << "[Puck-JNI] Query dimension: " << dimension << std::endl;
//         std::cerr << "[Puck-JNI] Query vector: [";
//         for (int i = 0; i < std::min(5, (int)dimension); i++) {
//             std::cerr << queryArray[i] << " ";
//         }
//         std::cerr << "...]" << std::endl;
        
//         // 4. Prepare request
//         puck::Request request;
//         request.topk = static_cast<uint32_t>(kJ);
//         request.feature = queryArray;
        
//         std::cerr << "[Puck-JNI] Request prepared, topk=" << request.topk << std::endl;
        
//         // 5. Prepare response
//         std::vector<float> distances(kJ);
//         std::vector<uint32_t> localIndices(kJ);
        
//         puck::Response response;
//         response.distance = distances.data();
//         response.local_idx = localIndices.data();
//         response.result_num = 0;
        
//         // 6. Execute search
//         std::cerr << "[Puck-JNI] Calling PuckIndex::search()..." << std::endl;
//         int ret = metadata->puckIndex->search(&request, &response);
        
//         std::cerr << "[Puck-JNI] Search returned: " << ret << std::endl;
//         std::cerr << "[Puck-JNI] Result count: " << response.result_num << std::endl;
        
//         if (ret != 0) {
//             std::cerr << "[Puck-JNI] ERROR: Search failed with code " << ret << std::endl;
//             throw std::runtime_error("PuckIndex::search failed: " + std::to_string(ret));
//         }
        
//         // 7. Create result array
//         int resultSize = std::min(kJ, static_cast<int>(response.result_num));
//         std::cerr << "[Puck-JNI] Creating result array, size=" << resultSize << std::endl;
        
//         jclass resultClass = jniUtil->FindClass(env, "org/opensearch/knn/index/query/KNNQueryResult");
//         jmethodID constructor = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");
        
//         jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);
        
//         // 8. Fill results
//         for (int i = 0; i < resultSize; i++) {
//             uint32_t localIdx = response.local_idx[i];
            
//             if (localIdx >= metadata->ids.size()) {
//                 std::cerr << "[Puck-JNI] ERROR: Invalid index " << localIdx 
//                           << " (max: " << metadata->ids.size() << ")" << std::endl;
//                 throw std::runtime_error("Invalid local index");
//             }
            
//             jlong docId = static_cast<jlong>(metadata->ids[localIdx]);
//             jfloat distance = response.distance[i];
            
//             std::cerr << "[Puck-JNI]   Result[" << i << "]: localIdx=" << localIdx 
//                       << " docId=" << docId << " dist=" << distance << std::endl;
            
//             jobject result = jniUtil->NewObject(env, resultClass, constructor, docId, distance);
//             jniUtil->SetObjectArrayElement(env, results, i, result);
//         }
        
//         std::cerr << "[Puck-JNI] ===== QueryIndex SUCCESS =====" << std::endl;
//         return results;
        
//     } catch (const std::exception& e) {
//         std::cerr << "[Puck-JNI] ===== QueryIndex ERROR: " << e.what() << " =====" << std::endl;
//         jniUtil->ThrowJavaException(env, "java/lang/RuntimeException", e.what());
//         return nullptr;
//     }
// }

// optimized version for qeuerying
// Rename the JNI method to indicate it returns raw arrays
jobjectArray QueryIndex(
    knn_jni::JNIUtilInterface* jniUtil,
    JNIEnv* env,
    jlong indexPointerJ,
    jfloatArray queryVectorJ,
    jint kJ,
    jobject methodParamsJ,
    jintArray parentIdsJ
) {
    try {
        // 1. Validate metadata
        auto* metadata = reinterpret_cast<IndexMetadata*>(indexPointerJ);
        if (!metadata || !metadata->puckIndex) {
            throw std::runtime_error("Invalid metadata or puckIndex pointer");
        }
        
        // 2. Get query vector using GetPrimitiveArrayCritical (fastest)
        float* queryArray = static_cast<float*>(
            jniUtil->GetPrimitiveArrayCritical(env, queryVectorJ, nullptr)
        );
        if (!queryArray) {
            throw std::runtime_error("Failed to get query vector");
        }
        
        // 3. Prepare request (no JNI calls in critical section)
        puck::Request request;
        request.topk = static_cast<uint32_t>(kJ);
        request.feature = queryArray;
        
        // 4. Prepare response buffers
        std::vector<float> distances(kJ);
        std::vector<uint32_t> localIndices(kJ);
        puck::Response response;
        response.distance = distances.data();
        response.local_idx = localIndices.data();
        response.result_num = 0;
        
        // 5. Execute search
        int ret = metadata->puckIndex->search(&request, &response);
        
        // 6. Release query vector immediately (end critical section)
        jniUtil->ReleasePrimitiveArrayCritical(env, queryVectorJ, queryArray, JNI_ABORT);
        
        if (ret != 0) {
            throw std::runtime_error("PuckIndex::search failed: " + std::to_string(ret));
        }
        
        int resultSize = std::min(kJ, static_cast<int>(response.result_num));
        
        // 7. Get cached class and method (already cached in JNIUtil::Initialize)
        jclass resultClass = jniUtil->FindClass(env, "org/opensearch/knn/index/query/KNNQueryResult");
        jmethodID constructor = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");
        
        // 8. Create result array
        jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);
        
        // 9. Pre-validate all indices (fail fast)
        for (int i = 0; i < resultSize; i++) {
            if (localIndices[i] >= metadata->ids.size()) {
                throw std::runtime_error("Invalid index: " + std::to_string(localIndices[i]) + 
                                        " at position " + std::to_string(i));
            }
        }
        
        // 10. Fill results with immediate local ref cleanup
        for (int i = 0; i < resultSize; i++) {
            uint32_t localIdx = localIndices[i];
            jint docId = static_cast<jint>(metadata->ids[localIdx]);
            jfloat distance = distances[i];
            
            // Create object using cached constructor
            jobject result = jniUtil->NewObject(env, resultClass, constructor, docId, distance);
            
            // Set in array
            jniUtil->SetObjectArrayElement(env, results, i, result);
            
            // Immediately delete local reference to reduce memory pressure
            jniUtil->DeleteLocalRef(env, result);
        }
        
        return results;
        
    } catch (const std::exception& e) {
        jniUtil->ThrowJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
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
        puck::IndexConf& conf = metadata->indexConf;
        
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
