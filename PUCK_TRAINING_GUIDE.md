# Puck Training Guide - Using Puck Like FAISS IVF

## Overview

Puck's hierarchical clustering method requires training, similar to FAISS IVF. This guide shows how to use Puck with the training API.

## Key Concepts

### Puck vs FAISS IVF Comparison

| Feature | FAISS IVF | Puck Hierarchical Cluster |
|---------|-----------|---------------------------|
| **Training Required** | ✅ Yes | ✅ Yes |
| **Clustering** | Single-level (nlist) | Two-level (coarse + fine) |
| **Compression** | Optional (PQ, SQ) | Built-in Product Quantization |
| **Memory Usage** | Varies by encoder | ~1/4 of original size |
| **Parameters** | nlist, nprobes | coarse_clusters, fine_clusters, pq_m, pq_nbits |

## Creating a Puck Index with Training

### Method 1: Using Model/Training API (Recommended - Same as FAISS IVF)

#### Step 1: Create a Training Index

```bash
POST /_plugins/_knn/models/puck_model/_train
{
  "training_index": "train-index",
  "training_field": "train_vector",
  "dimension": 128,
  "description": "Puck hierarchical cluster model",
  "method": {
    "name": "hierarchical_cluster",
    "engine": "puck",
    "space_type": "l2",
    "parameters": {
      "coarse_clusters": 256,
      "fine_clusters": 256,
      "pq_m": 8,
      "pq_nbits": 8
    }
  }
}
```

#### Step 2: Check Training Status

```bash
GET /_plugins/_knn/models/puck_model?filter_path=state

# Response:
{
  "state": "created"  # Wait until this becomes "training" then "created"
}
```

#### Step 3: Create Index Using Trained Model

```bash
PUT /my-puck-index
{
  "settings": {
    "index": {
      "knn": true,
      "number_of_shards": 1,
      "number_of_replicas": 0
    }
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "model_id": "puck_model"
      },
      "title": {
        "type": "text"
      }
    }
  }
}
```

### Method 2: Direct Index Creation (Training Happens During Segment Build)

**Note**: With `.setRequiresTraining(true)`, training will happen automatically during the first segment flush when enough vectors are collected.

```bash
PUT /puck-direct-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 128,
        "method": {
          "name": "hierarchical_cluster",
          "engine": "puck",
          "space_type": "l2",
          "parameters": {
            "coarse_clusters": 256,
            "fine_clusters": 256,
            "pq_m": 8,
            "pq_nbits": 8
          }
        }
      }
    }
  }
}
```

## Parameter Guidelines

### coarse_clusters
- **Description**: Number of first-level cluster centers
- **Default**: 256
- **Recommendation**: Similar to FAISS IVF's `nlist`
  - Small datasets (<100K vectors): 128-512
  - Medium datasets (100K-1M vectors): 512-2048
  - Large datasets (>1M vectors): 2048-16384
- **Formula**: `sqrt(num_vectors)` to `2 * sqrt(num_vectors)`

### fine_clusters
- **Description**: Number of second-level cluster centers per coarse cluster
- **Default**: 256
- **Recommendation**:
  - Balanced: 256
  - Higher accuracy: 512-1024
  - Lower memory: 128

### pq_m
- **Description**: Number of product quantization subspaces
- **Default**: 8
- **Recommendation**: Must divide dimension evenly
  - 768-dim: pq_m = 8, 16, 24, 32
  - 128-dim: pq_m = 8, 16
  - Higher values = better accuracy, more memory

### pq_nbits
- **Description**: Bits per PQ code
- **Default**: 8
- **Range**: 1-16
- **Recommendation**:
  - 8 bits (256 centers): Good balance
  - 16 bits (65536 centers): Higher accuracy, more memory
  - 4 bits (16 centers): Lower memory, lower accuracy

## Training Time Expectations

Training time depends on:
- Number of training vectors
- Number of dimensions
- Number of coarse/fine clusters
- PQ parameters

**Example timing for 100K vectors, 128-dim:**
- coarse_clusters=256, fine_clusters=256: ~2-5 minutes
- coarse_clusters=1024, fine_clusters=512: ~10-20 minutes

## Memory Usage

Puck uses approximately **1/4 of the original vector size** plus cluster overhead:

```
Memory ≈ (num_vectors * dimension * 4 bytes / 4)  # Compressed vectors
       + (coarse_clusters * dimension * 4)         # Coarse centroids
       + (coarse_clusters * fine_clusters * dimension * 4)  # Fine centroids
       + (pq_m * 256 * (dimension/pq_m) * 4)      # PQ codebooks
```

## Comparison with FAISS IVF

### FAISS IVF Index Creation:
```bash
PUT /faiss-ivf-index
{
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 128,
        "method": {
          "name": "ivf",
          "engine": "faiss",
          "space_type": "l2",
          "parameters": {
            "nlist": 256,
            "encoder": {
              "name": "pq",
              "parameters": {
                "m": 8,
                "code_size": 8
              }
            }
          }
        }
      }
    }
  }
}
```

### Puck Hierarchical Cluster (Equivalent):
```bash
PUT /puck-hierarchical-index
{
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 128,
        "method": {
          "name": "hierarchical_cluster",
          "engine": "puck",
          "space_type": "l2",
          "parameters": {
            "coarse_clusters": 256,    # Similar to IVF nlist
            "fine_clusters": 256,       # Additional refinement
            "pq_m": 8,                  # Similar to PQ m
            "pq_nbits": 8               # Similar to PQ code_size
          }
        }
      }
    }
  }
}
```

## Search API (Same as other engines)

```bash
GET /my-puck-index/_search
{
  "size": 10,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.1, 0.2, ...],  # 128-dimensional
        "k": 10
      }
    }
  }
}
```

## Benefits of Two-Level Hierarchical Clustering

1. **Better Accuracy**: Two-level refinement vs single-level IVF
2. **Memory Efficient**: Built-in compression to ~1/4 size
3. **Scalable**: Designed for billions of vectors
4. **Fast Search**: Hierarchical pruning reduces search space efficiently

## Troubleshooting

### Error: "Training required but not performed"
- **Solution**: Use the training API or ensure enough vectors for automatic training

### Slow indexing performance
- **Expected**: Training takes time (minutes to hours for large datasets)
- **Solution**: Use smaller cluster counts for faster training, or use the training API to train once and reuse

### High memory usage
- **Solution**: Reduce `pq_nbits` or `fine_clusters` parameters

## Next Steps

1. Start with default parameters for initial testing
2. Monitor training time and memory usage
3. Adjust parameters based on your accuracy/performance requirements
4. Compare with FAISS IVF for your specific use case

source /opt/intel/oneapi/setvars.sh && bash scripts/build.sh -v 2.19.3 -s false -o artifacts 

## set env
source /opt/intel/oneapi/mkl/2025.2/env/vars.sh
export CPATH=/usr/include/mkl:$CPATH
export C_INCLUDE_PATH=/usr/include/mkl:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/include/mkl:$CPLUS_INCLUDE_PATH

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
