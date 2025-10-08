# k-NN Plugin - 7 Main Modules Detailed Explanation (Corrected & Expanded)

> **Note:** This document is verified against the actual OpenSearch k-NN repository structure.

---

## 1. COMMON Module 🔧
**Path:** `src/main/java/org/opensearch/knn/common`

### Primary Function
Provides **shared constants and common utilities** used across all other modules. This is the foundation layer.

### Key Responsibilities
1. **Plugin-wide Constants**
   - File extensions for native indices
   - Default parameter values
   - Limits and thresholds
   - Index and field names

2. **Shared Utilities**
   - Field information extraction
   - Common helper functions
   - Type conversions

### Module Structure
```
common/
├── KNNConstants.java          # All plugin constants
└── FieldInfoExtractor.java    # Extract field metadata
```

### Example Constants
- `HNSW_EXTENSION = ".hnsw"` - File extension for HNSW graphs
- `MODEL_INDEX_NAME = ".opensearch-knn-models"` - System index name
- `DEFAULT_VECTOR_DATA_TYPE_FIELD` - Default vector data type
- Parameter defaults and limits

### Why This Module Exists
- **Single source of truth** for constants
- **Prevents code duplication** across modules
- **Easy maintenance** - change a constant in one place
- **No dependencies** - can be used by all modules

---

## 2. INDEX Module 📊
**Path:** `src/main/java/org/opensearch/knn/index`

### Primary Function
**Core indexing and search functionality** - the heart of the k-NN plugin. Handles everything from field mapping to query execution.

### Key Responsibilities

#### 2.1 Field Mapping (`mapper/`)
**Purpose:** Define how `knn_vector` fields are mapped and stored

```
mapper/
├── KNNVectorFieldMapper.java       # Main field mapper
├── KNNVectorFieldType.java         # Field type definition  
├── ModelFieldMapper.java           # Model-based field mapping
├── MethodFieldMapper.java          # Method configuration mapper
├── CompositeFieldMapper.java       # Composite field support
├── KNNMappingConfig.java           # Mapping configuration
├── ParameterValidator.java         # Validate field parameters
└── Mode.java                       # Indexing modes
```

**Responsibilities:**
- Parse `knn_vector` field definitions from index mappings
- Validate vector dimensions (1-16,000)
- Configure algorithms (HNSW, IVF)
- Select engines (Faiss, NMSLIB, Lucene)
- Handle model-based vs method-based fields
- Support nested and parent-child fields

**Example:** When you define a field like:
```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 768,
    "method": {
      "name": "hnsw",
      "engine": "faiss"
    }
  }
}
```
The mapper parses and validates this configuration.

#### 2.2 Custom Codec (`codec/`)
**Purpose:** Extend Lucene codec to support k-NN native indices

```
codec/
├── KNNCodecVersion.java            # Codec versioning
├── KNNCodecService.java            # Codec service registry
│
├── KNN990Codec/                    # Lucene 9.9.0 based
│   ├── KNN990Codec.java
│   └── KNN990PerFieldKnnVectorsFormat.java
│
├── KNN980Codec/                    # Lucene 9.8.0 based
│   ├── KNN980Codec.java
│   └── KNN980PerFieldKnnVectorsFormat.java
│
├── KNN950Codec/                    # Lucene 9.5.0 based
│   ├── KNN950Codec.java
│   └── KNN950PerFieldKnnVectorsFormat.java
│
├── nativeindex/                    # Native index I/O
│   ├── NativeIndexWriter.java      # Write .hnsw/.faiss files
│   ├── NativeIndexReader.java      # Read native indices
│   ├── DefaultIndexBuildSetup.java # Index building
│   └── MemoryOptimizedNativeIndexBuildStrategy.java
│
└── util/                           # Codec utilities
    ├── SerializationMode.java
    └── KNNCodecUtil.java
```

**Responsibilities:**
- Intercept Lucene's indexing pipeline
- Write vectors to native graph formats (.hnsw, .faiss)
- Read native indices during search
- Handle multiple codec versions
- Manage segment-level indices
- Support iterative graph building

**How It Works:**
1. During indexing, Lucene calls codec to store data
2. Codec writes vectors to both Lucene format AND native format
3. Native format (.hnsw/.faiss) stored alongside Lucene segments
4. During search, codec reads from native indices for fast ANN search

#### 2.3 Query Processing (`query/`)
**Purpose:** Build and execute k-NN queries

```
query/
├── KNNQueryBuilder.java            # Build k-NN queries from DSL
├── KNNQuery.java                   # Main query implementation
├── KNNWeight.java                  # Query weight for scoring
├── KNNScorer.java                  # Score documents
├── ExactSearcher.java              # Brute-force exact search
├── KNNQueryResult.java             # Query results wrapper
├── ResultUtil.java                 # Result utilities
└── rescore/                        # Rescoring support
    └── RescoreContext.java
```

**Responsibilities:**
- Parse k-NN query DSL
- Route to appropriate search method (ANN vs exact)
- Execute searches via JNI (for Faiss/NMSLIB)
- Execute Lucene-native searches
- Score and rank results
- Support post-filtering
- Handle rescoring for better accuracy

**Search Flow:**
```
Query DSL → QueryBuilder → Query → Weight → Scorer → Results
```

#### 2.4 Engine Management (`engine/`)
**Purpose:** Manage different k-NN algorithm engines

```
engine/
├── KNNEngine.java                  # Engine enum (Faiss/NMSLIB/Lucene)
├── AbstractKNNMethod.java          # Base method class
├── MethodComponent.java            # Method components
├── MethodComponentContext.java     # Component context
├── KNNMethodContext.java           # Method configuration
├── KNNMethodConfigContext.java     # Config context
├── EngineSpecificMethodContext.java
│
├── faiss/                          # Faiss engine
│   ├── FaissHNSWMethod.java        # HNSW implementation
│   ├── FaissIVFMethod.java         # IVF implementation
│   ├── AbstractFaissMethod.java    # Base Faiss method
│   └── FaissBinaryIndexBuildSetup.java
│
├── nmslib/                         # NMSLIB engine
│   └── NmslibHNSWMethod.java       # NMSW implementation
│
└── lucene/                         # Lucene engine
    └── LuceneHNSWMethod.java       # Lucene HNSW
```

**Responsibilities:**
- Abstract engine differences
- Configure algorithm parameters (M, ef_construction, ef_search, nlist)
- Validate engine-specific parameters
- Handle engine capabilities
- Support multiple algorithms per engine

**Engine Comparison:**
- **Lucene**: Native to OpenSearch, no JNI, easier management
- **NMSLIB**: Fast HNSW, more distance functions (L1, Linf)
- **Faiss**: Most features, quantization support, SIMD optimization

#### 2.5 Memory Management (`memory/`)
**Purpose:** Manage native memory for graph indices

```
memory/
├── NativeMemoryCacheManager.java   # Overall cache management
├── NativeMemoryAllocation.java     # Memory allocations
├── NativeMemoryEntryContext.java   # Entry metadata
├── NativeMemoryLoadStrategy.java   # Load strategies
└── SharedIndexStateManager.java    # Shared state
```

**Responsibilities:**
- Allocate native (off-heap) memory for graphs
- Cache loaded graphs for reuse
- Implement LRU eviction when memory is full
- Track memory usage per shard
- Coordinate with circuit breaker
- Handle concurrent access

**Memory Flow:**
```
Graph needed → Check cache → Not in cache → Load from disk
→ Allocate memory → Load graph → Add to cache → Use for search
```

#### 2.6 Settings & Configuration
**Purpose:** Manage all k-NN settings

```
KNNSettings.java                    # All settings definitions
IndexUtil.java                      # Index utilities
SpaceType.java                      # Distance functions enum
VectorDataType.java                 # Vector data types
```

**Responsibilities:**
- Define cluster and index settings
- Circuit breaker settings
- Cache size settings
- Algorithm parameters
- Distance functions (L2, Cosine, Inner Product, etc.)

### Sub-package Summary
| Sub-package | Purpose | Key Classes |
|-------------|---------|-------------|
| `mapper/` | Field mapping | KNNVectorFieldMapper |
| `codec/` | Lucene codec | KNN990Codec, NativeIndexWriter |
| `query/` | Query execution | KNNQueryBuilder, KNNScorer |
| `engine/` | Algorithm engines | KNNEngine, FaissHNSWMethod |
| `memory/` | Memory management | NativeMemoryCacheManager |
| `store/` | Storage utilities | IndexInputWithBuffer |
| `util/` | Index utilities | EngineResolver, FieldInfoExtractor |

---

## 3. INDICES Module 💾
**Path:** `src/main/java/org/opensearch/knn/indices`

### Primary Function
**Manage trained models** that can be shared across multiple indices. Handles the complete lifecycle from training to deletion.

### Key Responsibilities

```
indices/
├── Model.java                      # Model definition
├── ModelMetadata.java              # Model metadata
├── ModelDao.java                   # Model data access
├── ModelCache.java                 # In-memory model cache
├── ModelGraveyard.java             # Track deleted models
└── ModelUtil.java                  # Model utilities
```

#### 3.1 Model Storage
**Purpose:** Persist models in system index

**System Index:** `.opensearch-knn-models`

**What's Stored:**
- Model ID and description
- Algorithm method (IVF, PQ, etc.)
- Training data info (dimension, space type)
- Serialized model state (centroids, codebooks)
- Training timestamp and node

**Storage Flow:**
```
Train model → Serialize → Save to .opensearch-knn-models → Available cluster-wide
```

#### 3.2 Model CRUD Operations
**ModelDao.java** - Data Access Object

**Operations:**
- `put(Model)` - Save new model
- `get(modelId)` - Retrieve model
- `delete(modelId)` - Delete model
- `getMetadata(modelId)` - Get metadata only
- `search()` - List all models

**Access Pattern:**
```
User API Request → ModelDao → System Index → Return Model
```

#### 3.3 Model Caching
**ModelCache.java** - In-memory cache

**Purpose:** Cache frequently used models to avoid repeated deserialization

**Features:**
- LRU eviction when cache full
- Configurable cache size
- Thread-safe access
- Cache statistics

**Flow:**
```
Need model → Check cache → Hit: return → Miss: load from ModelDao → Add to cache → Return
```

#### 3.4 Model Lifecycle
**ModelGraveyard.java** - Track deleted models

**Purpose:** Prevent using models that are being deleted

**Features:**
- Mark models for deletion
- Prevent new usage of deleted models
- Clean up after all references released
- Coordinate with indices using the model

#### 3.5 When Models Are Used

**Training Required For:**
1. **IVF methods** - Need trained centroids
2. **PQ compression** - Need trained codebooks
3. **Combination methods** (IVF+PQ) - Need both

**Training NOT Required For:**
- HNSW methods (build incrementally)
- Flat indices
- SQ/Binary quantization (calculate from data)

**Example Use Case:**
```
1. Train IVF+PQ model on 100k vectors
2. Save model as "my-ivf-pq-model"
3. Use model in index1, index2, index3
4. All indices share same centroids/codebooks
5. Consistent compression across indices
```

---

## 4. JNI Module 🔗
**Path:** `src/main/java/org/opensearch/knn/jni` (Java) + `jni/` (C++)

### Primary Function
**Bridge Java and C++ native libraries** - enables calling Faiss and NMSLIB functions from Java code.

### Java Side

```
jni/
├── JNIService.java                 # Main JNI interface
├── JNICommons.java                 # Common JNI utilities
└── PlatformSpecificInfo.java       # Platform detection
```

#### 4.1 JNI Interface
**JNIService.java** - Main interface

**Native Methods:**
```java
// Faiss methods
public static native long createIndex(...)
public static native void insertToIndex(...)
public static native long[] queryIndex(...)
public static native void deleteIndex(...)

// NMSLIB methods  
public static native long createIndexFromTemplate(...)
public static native void insertToIndexFromTemplate(...)
public static native long[] queryIndexFromTemplate(...)

// Common methods
public static native void freeSharedIndexState(...)
```

**Responsibilities:**
- Declare native methods (implemented in C++)
- Handle data marshalling (Java ↔ C++)
- Load native libraries
- Manage native memory references
- Handle exceptions from native code

#### 4.2 Library Loading
**Platform Detection:**
- Detect OS (Linux, macOS, Windows)
- Detect architecture (x86_64, ARM64)
- Check CPU features (AVX2, AVX512, Neon)

**Libraries Loaded:**
```
Linux x86_64 with AVX2:
- libopensearchknn_common.so
- libopensearchknn_faiss_avx2.so
- libopensearchknn_nmslib.so

Linux ARM64:
- libopensearchknn_common.so
- libopensearchknn_faiss.so (with Neon)
- libopensearchknn_nmslib.so
```

### C++ Side

```
jni/
├── include/
│   ├── faiss_wrapper.h             # Faiss JNI header
│   ├── nmslib_wrapper.h            # NMSLIB JNI header
│   ├── jni_util.h                  # JNI utilities
│   └── commons.h                   # Common definitions
│
├── src/
│   ├── faiss_wrapper.cpp           # Faiss implementation
│   ├── nmslib_wrapper.cpp          # NMSLIB implementation
│   ├── jni_util.cpp                # JNI utilities
│   ├── org_opensearch_knn_jni_FaissService.cpp
│   └── org_opensearch_knn_jni_NmslibService.cpp
│
├── external/
│   ├── faiss/                      # Faiss library
│   └── nmslib/                     # NMSLIB library
│
└── release/
    ├── libopensearchknn_common.so
    ├── libopensearchknn_faiss.so
    ├── libopensearchknn_faiss_avx2.so
    └── libopensearchknn_nmslib.so
```

#### 4.3 Data Marshalling

**Java to C++:**
- `float[]` → `float*`
- `long` (index pointer) → `void*`
- `String` → `const char*`
- Objects → Serialized bytes

**C++ to Java:**
- Results array → `long[]`
- Distances → `float[]`
- Index pointer → `long` (handle)

**Example Call Stack:**
```
Java: KNNScorer.score()
  ↓
Java: JNIService.queryIndex(indexPtr, vector, k)
  ↓ [JNI Boundary]
C++: org_opensearch_knn_jni_FaissService_queryIndex()
  ↓
C++: knn_jni::faiss_wrapper::QueryIndex()
  ↓
C++: faiss::Index::search()
  ↓ [Return path]
Java: long[] results
```

#### 4.4 Memory Management

**Critical Concept:** Native memory is **outside JVM heap**

**Challenges:**
- Java GC doesn't track native memory
- Must manually free native allocations
- Memory leaks possible if not careful

**Solution:**
- Track native pointers as Java `long`
- Explicit cleanup methods
- Circuit breaker monitors total usage
- Cache manager coordinates releases

### Build System

**CMakeLists.txt** - Build configuration

**Build Process:**
```bash
cd jni
cmake .  # Configure build
make opensearchknn_faiss opensearchknn_nmslib  # Build libraries
```

**Output:** Libraries in `jni/release/`

**Dependencies:**
- OpenMP (parallelization)
- BLAS/LAPACK (linear algebra)
- Faiss library
- NMSLIB library

---

## 5. PLUGIN Module 🔌
**Path:** `src/main/java/org/opensearch/knn/plugin`

### Primary Function
**Integrate k-NN with OpenSearch** - main entry point that registers all components and exposes plugin functionality.

### Main Plugin Class

```
plugin/
└── KNNPlugin.java                  # Main plugin entry point
```

**KNNPlugin.java** implements:
- `ActionPlugin` - Register transport actions
- `MapperPlugin` - Register field mapper
- `SearchPlugin` - Register query builder
- `EnginePlugin` - Register codec
- `ScriptPlugin` - Register script engine

**Registration Methods:**
```java
// Register knn_vector field type
@Override
public Map<String, Mapper.TypeParser> getMappers()

// Register KNN codecs
@Override
public Map<String, Codec> getCodecs()

// Register knn query
@Override
public List<QuerySpec<?>> getQueries()

// Register REST handlers
@Override
public List<RestHandler> getRestHandlers()

// Register transport actions
@Override
public List<ActionHandler<?, ?>> getActions()

// Register script engine
@Override
public ScriptEngine getScriptEngine(...)
```

### Sub-packages

#### 5.1 REST API (`rest/`)
**Purpose:** HTTP endpoint handlers

```
rest/
├── RestKNNStatsHandler.java        # GET /_plugins/_knn/stats
├── RestKNNWarmupHandler.java       # GET /_plugins/_knn/warmup
├── RestGetModelHandler.java        # GET /_plugins/_knn/models/{id}
├── RestDeleteModelHandler.java     # DELETE /_plugins/_knn/models/{id}
├── RestSearchModelHandler.java     # GET /_plugins/_knn/models/_search  
├── RestTrainModelHandler.java      # POST /_plugins/_knn/models/{id}/_train
└── RestClearCacheHandler.java      # POST /_plugins/_knn/cache/clear
```

**Responsibilities:**
- Parse REST requests
- Validate parameters
- Call transport actions
- Format responses
- Handle errors

**Example Request:**
```
POST /_plugins/_knn/models/my-model/_train
{
  "training_index": "train-data",
  "training_field": "my_vector",
  "dimension": 768,
  "method": {
    "name": "ivf",
    "engine": "faiss",
    "parameters": {
      "nlist": 128
    }
  }
}
```

#### 5.2 Transport Layer (`transport/`)
**Purpose:** Inter-node communication

```
transport/
├── KNNWarmupTransportAction.java   # Warmup graphs
├── KNNStatsTransportAction.java    # Collect stats
├── GetModelTransportAction.java    # Get model
├── DeleteModelTransportAction.java # Delete model
├── TrainingJobRouterTransportAction.java  # Route training
├── TrainingModelTransportAction.java      # Execute training
├── UpdateModelMetadataTransportAction.java
└── UpdateModelGraveyardTransportAction.java
```

**Responsibilities:**
- Coordinate across cluster nodes
- Broadcast operations (warmup, stats)
- Route to specific nodes (training)
- Aggregate results
- Handle node failures

**Example: Warmup Action**
```
User requests warmup
  ↓
REST handler receives request
  ↓
Calls WarmupTransportAction
  ↓
Broadcast to all data nodes
  ↓
Each node loads graphs into memory
  ↓
Aggregate responses
  ↓
Return to user
```

#### 5.3 Script Scoring (`script/`)
**Purpose:** Enable k-NN in script_score queries

```
script/
├── KNNScoringScriptEngine.java     # Script engine
├── KNNScoreScript.java             # Score script base
└── KNNWhitelistExtension.java      # Painless whitelist
```

**Responsibilities:**
- Provide k-NN scoring in Painless scripts
- Enable exact k-NN search
- Support custom distance functions
- Allow pre-filtering

**Example Usage:**
```json
{
  "query": {
    "script_score": {
      "query": {"match_all": {}},
      "script": {
        "source": "knn_score",
        "lang": "knn",
        "params": {
          "field": "my_vector",
          "query_value": [1.0, 2.0, 3.0],
          "space_type": "cosinesimil"
        }
      }
    }
  }
}
```

#### 5.4 Statistics & Circuit Breaker (`stats/`)
**Purpose:** Monitor and protect the plugin

```
stats/
├── KNNCircuitBreaker.java          # Circuit breaker
├── KNNStats.java                   # Statistics service
├── KNNCounter.java                 # Counters
├── KNNStat.java                    # Stat definition
└── StatNames.java                  # Stat name constants
```

**Circuit Breaker:**
- Monitors native memory usage
- Triggers when memory exceeds limit
- Prevents new graph loading
- Forces eviction of cached graphs
- Protects from OOM

**Statistics Collected:**
- `circuit_breaker_triggered` - CB trigger count
- `graph_memory_usage` - Native memory used
- `graph_index_requests` - Index operations
- `graph_query_requests` - Query operations
- `cache_capacity_reached` - Cache full count
- `script_query_requests` - Script query count
- `script_query_errors` - Script query errors

**Stat Types:**
- Counter: Monotonically increasing
- Gauge: Current value
- Timer: Duration measurements

---

## 6. QUANTIZATION Module 📦
**Path:** `src/main/java/org/opensearch/knn/quantization`

### Primary Function
**Compress vectors** to reduce memory footprint while maintaining acceptable search accuracy.

### Module Structure

```
quantization/
├── quantizer/                      # Quantizer implementations
├── models/                         # Quantization models
├── sampler/                        # Data sampling
└── enums/                          # Quantization enums
```

### Sub-packages

#### 6.1 Quantizers (`quantizer/`)
**Purpose:** Implement quantization algorithms

```
quantizer/
├── Quantizer.java                  # Base quantizer interface
├── QuantizerRegistry.java          # Registry of quantizers
├── MultiBitScalarQuantizer.java   # Multi-bit SQ
├── OneBitScalarQuantizer.java     # 1-bit binary
└── QuantizationConfig.java         # Configuration
```

**Quantizer Types:**

**1. Product Quantization (PQ)**
- Divide vector into m sub-vectors
- Quantize each sub-vector independently
- Use codebooks (256 entries per sub-vector)
- 8-32x compression
- Supported by Faiss engine only

**2. Scalar Quantization (SQ)**
- Convert float32 → int8/int16
- Learn min/max per dimension
- 2-4x compression
- Fast operations
- Supported by Faiss and Lucene

**3. Binary Quantization**
- Convert to 1-bit (binary)
- 32x compression
- Hamming distance
- Maximum compression, lower accuracy

**4. FP16 Quantization**
- Convert float32 → float16
- 2x compression
- Minimal accuracy loss
- Hardware accelerated

**Quantizer Selection:**
```
Need 8x+ compression → PQ
Need 4x compression, fast → SQ
Need max compression → Binary
Need minimal accuracy loss → FP16
```

#### 6.2 Quantization Models (`models/`)
**Purpose:** Store quantization state and parameters

```
models/
├── quantizationState/
│   ├── QuantizationState.java                # Base state
│   ├── QuantizationStateCache.java           # State cache
│   ├── OneBitScalarQuantizationState.java    # Binary state
│   └── MultiBitScalarQuantizationState.java  # SQ state
│
└── quantizationParams/
    ├── QuantizationParams.java               # Base params
    └── ScalarQuantizationParams.java         # SQ params
```

**Quantization State Contains:**
- Codebooks (for PQ)
- Min/max values (for SQ)
- Quantization type
- Number of bits
- Trained or calculated

**State Cache:**
- Cache frequently used states
- Avoid recalculating
- LRU eviction
- Per-index state

#### 6.3 Sampling (`sampler/`)
**Purpose:** Sample vectors for training quantizers

```
sampler/
├── Sampler.java                    # Base sampler
├── ReservoirSampler.java           # Reservoir sampling
└── SamplingFactory.java            # Create samplers
```

**Sampling Methods:**
- **Reservoir Sampling**: Fixed-size sample from stream
- Used for training PQ codebooks
- Ensures representative sample

### Quantization Workflow

#### Training Phase (for PQ):
```
1. Sample vectors from index
2. Train quantizer (learn codebooks)
3. Save quantization state
4. Cache state for use
```

#### Indexing Phase:
```
1. Get quantization state
2. Compress vector using quantizer
3. Store compressed vector
4. Native index uses compressed vectors
```

#### Search Phase:
```
1. Compress query vector
2. Search in compressed space
3. Calculate approximate distances
4. Optional: Rescore with full vectors
5. Return results
```

### Memory Savings Example

**Original Vector (768D float32):**
- Size: 768 × 4 bytes = 3,072 bytes

**Quantization Options:**
- **FP16**: 768 × 2 = 1,536 bytes (2x smaller)
- **SQ8**: 768 × 1 = 768 bytes (4x smaller)  
- **PQ (m=96, 8-bit)**: 96 bytes (32x smaller)
- **Binary**: 768 ÷ 8 = 96 bytes (32x smaller)

**For 1 million vectors:**
- Original: 3 GB
- FP16: 1.5 GB
- SQ8: 768 MB
- PQ: 96 MB
- Binary: 96 MB

### Trade-offs

**Higher Compression = More Memory Savings but Lower Accuracy**

| Method | Compression | Accuracy | Speed | Use Case |
|--------|-------------|----------|-------|----------|
| FP16 | 2x | 99%+ | Fast | General purpose |
| SQ8 | 4x | 95-98% | Fast | Good balance |
| PQ | 8-32x | 90-95% | Medium | Memory limited |
| Binary | 32x | 80-90% | Very Fast | Extreme scale |

---

## 7. TRAINING Module 🎓
**Path:** `src/main/java/org/opensearch/knn/training`

### Primary Function
**Coordinate model training** - manages the distributed process of training k-NN models (primarily for IVF and PQ algorithms).

### Module Structure

```
training/
├── TrainingJobRunner.java          # Execute training jobs
├── TrainingDataConsumer.java       # Consume training data
├── VectorReader.java               # Read vectors
├── TrainingJob.java                # Training job definition
└── TrainingDataAllocation.java     # Data allocation
```

### Key Components

#### 7.1 Training Job Lifecycle

**TrainingJob.java** - Job definition
```java
class TrainingJob {
    String modelId;
    String trainingIndex;
    String trainingField;
    KNNMethodContext methodContext;
    int dimension;
    String description;
    // ...
}
```

**Phases:**
```
1. Job Creation
   ↓
2. Data Sampling
   ↓
3. Data Collection
   ↓
4. Model Training
   ↓
5. Model Persistence
   ↓
6. Completion
```

#### 7.2 Training Coordination

**TrainingJobRunner.java** - Main coordinator

**Responsibilities:**

#### 7.3 Data Collection

**VectorReader.java** - Read training vectors

**Purpose:** Efficiently read vectors from source index

**Methods:**
- `read()` - Read batch of vectors
- `getTotalLiveDocs()` - Get document count
- `getDimension()` - Get vector dimension

**Optimization:**
- Stream data (don't load all into memory)
- Read in batches
- Skip deleted documents
- Handle multiple segments

**TrainingDataConsumer.java** - Consume and aggregate data

**Purpose:** Collect vectors for training

**Process:**
```
1. Determine sample size
2. Use ReservoirSampler
3. Collect from multiple shards
4. Aggregate on coordinator node
5. Pass to training algorithm
```

#### 7.4 Training Execution

**Native Training via JNI:**

**For IVF:**
```java
// Sample 256 * nlist vectors
// Train k-means clustering
// Learn nlist centroids
// Return trained index
```

**For PQ:**
```java
// Sample vectors
// Divide into sub-vectors
// Train codebooks (256 entries per sub-vector)
// Return trained quantizer
```

**For IVF+PQ:**
```java
// First train IVF (centroids)
// Then train PQ (codebooks)
// Return combined trained model
```

#### 7.5 Model Persistence

**After Training:**
```
1. Serialize model state
2. Create Model object
3. Create ModelMetadata
4. Save via ModelDao
5. Add to ModelCache
6. Make available cluster-wide
```

**Model Contents:**
- Algorithm configuration
- Training parameters
- Trained state (centroids/codebooks)
- Dimension and space type
- Training statistics

### When Training Is Required

**Algorithms Requiring Training:**

| Algorithm | What's Trained | Sample Size |
|-----------|----------------|-------------|
| IVF | Centroids (cluster centers) | 256 × nlist vectors |
| PQ | Codebooks (quantization tables) | 10,000+ vectors |
| IVF+PQ | Both centroids and codebooks | 256 × nlist vectors |

**Algorithms NOT Requiring Training:**
- HNSW (builds incrementally during indexing)
- Flat indices (no preprocessing)
- SQ (calculates from actual data)
- Binary (thresholds calculated)

### Training Best Practices

**Sample Size:**
- **IVF**: At least 256 × nlist vectors
  - For nlist=128: need 32,768+ vectors
- **PQ**: At least 10,000 vectors
  - More samples = better codebooks

**Training Index:**
- Use representative data
- Same distribution as search data
- Sufficient volume
- Can be same as target index or separate

**Resource Usage:**
- Training is CPU intensive
- Runs on single node
- Can take minutes to hours
- Uses JNI/native memory

**Example Training Request:**
```json
POST /_plugins/_knn/models/my-ivf-pq-model/_train
{
  "training_index": "training-vectors",
  "training_field": "my_vector",
  "dimension": 768,
  "description": "IVF+PQ model for 768D vectors",
  "method": {
    "name": "ivf",
    "engine": "faiss",
    "space_type": "l2",
    "parameters": {
      "nlist": 128,
      "encoder": {
        "name": "pq",
        "parameters": {
          "m": 96,
          "code_size": 8
        }
      }
    }
  }
}
```

**Response:**
```json
{
  "model_id": "my-ivf-pq-model",
  "state": "created"
}
```

---

## Module Dependencies Deep Dive

### Dependency Graph

```
                    ┌─────────────┐
                    │   PLUGIN    │ ◄─── Entry Point
                    │  (Registers │      Coordinates all modules
                    │    all)     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┬──────────────┐
         │                 │                 │              │
         ▼                 ▼                 ▼              ▼
    ┌────────┐        ┌────────┐       ┌──────────┐   ┌──────────┐
    │ INDEX  │        │INDICES │       │ TRAINING │   │  COMMON  │
    │(Core)  │        │(Models)│       │  (Jobs)  │   │(Shared)  │
    └───┬────┘        └────────┘       └────┬─────┘   └─────▲────┘
        │                                   │               │
        │              ┌────────────────────┘               │
        │              │                                    │
        ▼              ▼                                    │
    ┌────────┐    ┌────────┐                               │
    │  JNI   │    │QUANT-  │                               │
    │(Bridge)│    │IZATION │───────────────────────────────┘
    └────────┘    └────────┘
        │
        ▼
    [Native Libraries]
    - Faiss
    - NMSLIB
```

### Import Dependencies

**COMMON Module:**
- **Imports:** None (foundation layer)
- **Used by:** All other modules

**INDEX Module:**
- **Imports:** COMMON, JNI, QUANTIZATION
- **Uses:**
  - `JNIService` for native calls
  - `Quantizer` for compression
  - `KNNConstants` for constants

**INDICES Module:**
- **Imports:** COMMON
- **Used by:** PLUGIN, TRAINING
- **Purpose:** Isolated model management

**JNI Module:**
- **Imports:** COMMON
- **Used by:** INDEX, QUANTIZATION, TRAINING
- **Purpose:** Bridge to native code

**PLUGIN Module:**
- **Imports:** All modules
- **Purpose:** Orchestration and registration

**QUANTIZATION Module:**
- **Imports:** COMMON, JNI
- **Used by:** INDEX
- **Purpose:** Vector compression

**TRAINING Module:**
- **Imports:** COMMON, INDICES, JNI
- **Used by:** PLUGIN (via REST/Transport)
- **Purpose:** Model training coordination

### Cross-Module Communication Patterns

#### Pattern 1: Query Execution
```
PLUGIN (REST) 
  → INDEX (QueryBuilder) 
    → INDEX (Query/Scorer) 
      → JNI (Native call)
        → [Faiss/NMSLIB]
```

#### Pattern 2: Model Training
```
PLUGIN (REST) 
  → TRAINING (JobRunner) 
    → TRAINING (VectorReader) 
      → JNI (Train call)
        → [Faiss training]
      → INDICES (ModelDao.save)
```

#### Pattern 3: Indexing with Quantization
```
INDEX (Codec) 
  → QUANTIZATION (Quantizer) 
    → QUANTIZATION (State cache) 
      → INDEX (Compressed vector) 
        → JNI (Write to native)
```

#### Pattern 4: Warmup Operation
```
PLUGIN (REST) 
  → PLUGIN (Transport) 
    → INDEX (Memory Manager) 
      → JNI (Load graphs)
```

---

## File Organization Summary

### Complete Package Structure

```
src/main/java/org/opensearch/knn/
│
├── common/                          # Module 1: COMMON
│   ├── KNNConstants.java
│   └── FieldInfoExtractor.java
│
├── index/                           # Module 2: INDEX
│   ├── KNNSettings.java
│   ├── SpaceType.java
│   ├── VectorDataType.java
│   ├── IndexUtil.java
│   │
│   ├── mapper/                      # Field mapping
│   │   ├── KNNVectorFieldMapper.java
│   │   ├── KNNVectorFieldType.java
│   │   ├── ModelFieldMapper.java
│   │   ├── MethodFieldMapper.java
│   │   ├── CompositeFieldMapper.java
│   │   ├── KNNMappingConfig.java
│   │   ├── ParameterValidator.java
│   │   └── Mode.java
│   │
│   ├── codec/                       # Lucene codec
│   │   ├── KNNCodecVersion.java
│   │   ├── KNNCodecService.java
│   │   ├── KNN990Codec/
│   │   ├── KNN980Codec/
│   │   ├── KNN950Codec/
│   │   ├── nativeindex/
│   │   └── util/
│   │
│   ├── query/                       # Query processing
│   │   ├── KNNQueryBuilder.java
│   │   ├── KNNQuery.java
│   │   ├── KNNWeight.java
│   │   ├── KNNScorer.java
│   │   ├── ExactSearcher.java
│   │   ├── KNNQueryResult.java
│   │   ├── ResultUtil.java
│   │   └── rescore/
│   │
│   ├── engine/                      # Algorithm engines
│   │   ├── KNNEngine.java
│   │   ├── AbstractKNNMethod.java
│   │   ├── MethodComponent.java
│   │   ├── KNNMethodContext.java
│   │   ├── faiss/
│   │   ├── nmslib/
│   │   └── lucene/
│   │
│   ├── memory/                      # Memory management
│   │   ├── NativeMemoryCacheManager.java
│   │   ├── NativeMemoryAllocation.java
│   │   ├── NativeMemoryEntryContext.java
│   │   └── NativeMemoryLoadStrategy.java
│   │
│   ├── store/                       # Storage utilities
│   │   └── IndexInputWithBuffer.java
│   │
│   └── util/                        # Index utilities
│       ├── EngineResolver.java
│       └── FieldInfoExtractor.java
│
├── indices/                         # Module 3: INDICES
│   ├── Model.java
│   ├── ModelMetadata.java
│   ├── ModelDao.java
│   ├── ModelCache.java
│   ├── ModelGraveyard.java
│   └── ModelUtil.java
│
├── jni/                            # Module 4: JNI (Java side)
│   ├── JNIService.java
│   ├── JNICommons.java
│   └── PlatformSpecificInfo.java
│
├── plugin/                          # Module 5: PLUGIN
│   ├── KNNPlugin.java              # Main entry
│   │
│   ├── rest/                        # REST handlers
│   │   ├── RestKNNStatsHandler.java
│   │   ├── RestKNNWarmupHandler.java
│   │   ├── RestGetModelHandler.java
│   │   ├── RestDeleteModelHandler.java
│   │   ├── RestSearchModelHandler.java
│   │   ├── RestTrainModelHandler.java
│   │   └── RestClearCacheHandler.java
│   │
│   ├── transport/                   # Transport actions
│   │   ├── KNNWarmupTransportAction.java
│   │   ├── KNNStatsTransportAction.java
│   │   ├── GetModelTransportAction.java
│   │   ├── DeleteModelTransportAction.java
│   │   ├── TrainingJobRouterTransportAction.java
│   │   ├── TrainingModelTransportAction.java
│   │   ├── UpdateModelMetadataTransportAction.java
│   │   └── UpdateModelGraveyardTransportAction.java
│   │
│   ├── script/                      # Script scoring
│   │   ├── KNNScoringScriptEngine.java
│   │   ├── KNNScoreScript.java
│   │   └── KNNWhitelistExtension.java
│   │
│   └── stats/                       # Statistics
│       ├── KNNCircuitBreaker.java
│       ├── KNNStats.java
│       ├── KNNCounter.java
│       ├── KNNStat.java
│       └── StatNames.java
│
├── quantization/                    # Module 6: QUANTIZATION
│   ├── quantizer/
│   │   ├── Quantizer.java
│   │   ├── QuantizerRegistry.java
│   │   ├── MultiBitScalarQuantizer.java
│   │   ├── OneBitScalarQuantizer.java
│   │   └── QuantizationConfig.java
│   │
│   ├── models/
│   │   ├── quantizationState/
│   │   │   ├── QuantizationState.java
│   │   │   ├── QuantizationStateCache.java
│   │   │   ├── OneBitScalarQuantizationState.java
│   │   │   └── MultiBitScalarQuantizationState.java
│   │   └── quantizationParams/
│   │       ├── QuantizationParams.java
│   │       └── ScalarQuantizationParams.java
│   │
│   ├── sampler/
│   │   ├── Sampler.java
│   │   ├── ReservoirSampler.java
│   │   └── SamplingFactory.java
│   │
│   └── enums/
│       └── ScalarQuantizationType.java
│
└── training/                        # Module 7: TRAINING
    ├── TrainingJobRunner.java
    ├── TrainingDataConsumer.java
    ├── VectorReader.java
    ├── TrainingJob.java
    └── TrainingDataAllocation.java
```

### C++ Code Structure

```
jni/
├── CMakeLists.txt                   # Build configuration
│
├── include/                         # Headers
│   ├── faiss_wrapper.h
│   ├── nmslib_wrapper.h
│   ├── jni_util.h
│   └── commons.h
│
├── src/                            # Implementation
│   ├── faiss_wrapper.cpp
│   ├── nmslib_wrapper.cpp
│   ├── jni_util.cpp
│   ├── org_opensearch_knn_jni_FaissService.cpp
│   └── org_opensearch_knn_jni_NmslibService.cpp
│
├── external/                        # External libraries
│   ├── faiss/
│   └── nmslib/
│
└── release/                         # Build output
    ├── libopensearchknn_common.so
    ├── libopensearchknn_faiss.so
    ├── libopensearchknn_faiss_avx2.so
    └── libopensearchknn_nmslib.so
```

---

## Practical Examples

### Example 1: Creating a k-NN Index

**Modules Involved:** PLUGIN → INDEX (mapper, codec)

```json
PUT /my-vectors
{
  "settings": {
    "index.knn": true,
    "number_of_shards": 3
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "engine": "faiss",
          "space_type": "l2",
          "parameters": {
            "m": 16,
            "ef_construction": 128
          }
        }
      }
    }
  }
}
```

**Internal Flow:**
1. REST API → INDEX.KNNVectorFieldMapper parses field
2. INDEX.ParameterValidator validates parameters
3. INDEX.KNNCodecService registers codec
4. INDEX.KNN990Codec handles writes
5. JNI.JNIService loads native libraries

### Example 2: Searching Vectors

**Modules Involved:** PLUGIN → INDEX (query) → JNI

```json
GET /my-vectors/_search
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.1, 0.2, ...],
        "k": 10
      }
    }
  }
}
```

**Internal Flow:**
1. REST API → INDEX.KNNQueryBuilder builds query
2. INDEX.KNNQuery executes search
3. INDEX.KNNWeight calculates weights
4. INDEX.KNNScorer scores documents
5. JNI.JNIService.queryIndex() calls Faiss
6. Results returned and ranked

### Example 3: Training a Model

**Modules Involved:** PLUGIN → TRAINING → INDICES → JNI

```json
POST /_plugins/_knn/models/my-model/_train
{
  "training_index": "train-vectors",
  "training_field": "my_vector",
  "dimension": 768,
  "method": {
    "name": "ivf",
    "engine": "faiss",
    "parameters": {
      "nlist": 128
    }
  }
}
```

**Internal Flow:**
1. PLUGIN.RestTrainModelHandler receives request
2. PLUGIN.TrainingJobRouterTransportAction routes job
3. TRAINING.TrainingJobRunner starts training
4. TRAINING.VectorReader samples vectors
5. JNI.JNIService trains IVF model
6. INDICES.ModelDao saves model
7. INDICES.ModelCache caches model

### Example 4: Using Quantization

**Modules Involved:** INDEX → QUANTIZATION → JNI

```json
PUT /compressed-vectors
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "engine": "faiss",
          "parameters": {
            "encoder": {
              "name": "sq",
              "parameters": {
                "type": "fp16"
              }
            }
          }
        }
      }
    }
  }
}
```

**Internal Flow:**
1. INDEX.KNNVectorFieldMapper parses encoder
2. QUANTIZATION.QuantizerRegistry gets SQ quantizer
3. During indexing: QUANTIZATION compresses vectors
4. INDEX.Codec writes compressed vectors
5. JNI stores in native format
6. Search uses compressed space

---

## Summary Table

| Module | Lines of Code* | Primary Purpose | Key Classes |
|--------|---------------|-----------------|-------------|
| **COMMON** | ~500 | Constants & utilities | KNNConstants |
| **INDEX** | ~15,000 | Core indexing/search | KNNVectorFieldMapper, KNN990Codec, KNNQuery |
| **INDICES** | ~2,000 | Model management | Model, ModelDao, ModelCache |
| **JNI** | ~3,000 (Java)<br>~5,000 (C++) | Java↔C++ bridge | JNIService |
| **PLUGIN** | ~5,000 | OpenSearch integration | KNNPlugin, REST handlers |
| **QUANTIZATION** | ~3,000 | Vector compression | Quantizer, QuantizationState |
| **TRAINING** | ~1,500 | Model training | TrainingJobRunner |

*Approximate values

**Total:** ~35,000 lines of Java + ~5,000 lines of C++

