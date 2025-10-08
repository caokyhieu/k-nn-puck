# Puck Index Testing Examples

This directory contains Python scripts for testing and evaluating the Puck hierarchical cluster index implementation in OpenSearch k-NN.

## Requirements

```bash
pip install opensearch-py numpy scikit-learn
```

## Quick Start Example

The quick start example demonstrates the complete workflow:

```bash
python examples/puck_quickstart.py
```

This script will:
1. Create a training index with 10,000 vectors
2. Train a Puck model with hierarchical clustering
3. Create an index using the trained model
4. Index 50,000 vectors
5. Run searches and measure performance

**Expected output:**
```
Training completed! Model size: 2.5 MB
Indexed 50000 vectors in 45.32 seconds
Average latency: 12.45 ms
P95 latency: 18.23 ms
QPS: 80.32
```

## Accuracy Testing Script

The full accuracy testing script (`test_puck_accuracy.py`) provides comprehensive evaluation:

```bash
# Basic test with default parameters
python test_puck_accuracy.py --host localhost --port 9200

# Custom configuration
python test_puck_accuracy.py \
    --host localhost \
    --port 9200 \
    --num-train 10000 \
    --num-database 100000 \
    --num-queries 1000 \
    --dimension 128 \
    --k 10 \
    --coarse-clusters 512 \
    --fine-clusters 256 \
    --pq-m 8 \
    --pq-nbits 8
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | localhost | OpenSearch host |
| `--port` | 9200 | OpenSearch port |
| `--num-train` | 10000 | Number of training vectors |
| `--num-database` | 50000 | Number of database vectors to index |
| `--num-queries` | 100 | Number of query vectors for testing |
| `--dimension` | 128 | Vector dimension |
| `--k` | 10 | K for top-K search |
| `--coarse-clusters` | 256 | Number of coarse clusters |
| `--fine-clusters` | 256 | Number of fine clusters |
| `--pq-m` | 8 | PQ subspaces (must divide dimension) |
| `--pq-nbits` | 8 | PQ bits (1-16) |
| `--use-ssl` | False | Use SSL connection |

### Metrics Measured

The accuracy testing script measures:

- **Recall@K**: Percentage of true nearest neighbors found in top-K results
- **Precision@K**: Percentage of returned results that are true nearest neighbors
- **Training Time**: Time to train the model
- **Model Size**: Size of the trained model blob
- **Indexing Time**: Time to index all vectors
- **Indexing Throughput**: Vectors indexed per second
- **Search Latency**: Average, P50, P95, P99 latencies
- **QPS**: Queries per second

### Example Output

```
================================================================================
PUCK ACCURACY AND PERFORMANCE TEST
================================================================================
Configuration:
  Training vectors: 10000
  Database vectors: 50000
  Query vectors: 100
  Dimension: 128
  K: 10
  Coarse clusters: 256
  Fine clusters: 256
  PQ subspaces (m): 8
  PQ bits: 8
================================================================================

Step 1: Generating vectors...
✓ Generated 10000 training, 50000 database, 100 query vectors

Step 2: Creating training index...
✓ Created training index: puck_train_index
✓ Indexed 10000 training vectors

Step 3: Training Puck model...
✓ Training completed in 45.32 seconds

Step 4: Creating index with trained model...
✓ Created index with trained model: puck_test_index

Step 5: Indexing database vectors...
✓ Indexed 50000 vectors in 67.89 seconds
  Throughput: 736.45 docs/sec

Step 6: Computing ground truth...
✓ Ground truth computed

Step 7: Running queries and measuring accuracy...
✓ Completed 100 queries

================================================================================
RESULTS
================================================================================
Training Time: 45.32 seconds
Model Size: 2.45 MB
Indexing Time: 67.89 seconds
Indexing Throughput: 736.45 docs/sec

Average Recall@10: 0.9234 (92.34%)
Average Precision@10: 0.9234 (92.34%)

Average Search Latency: 12.45 ms
P50 Search Latency: 11.23 ms
P95 Search Latency: 18.67 ms
P99 Search Latency: 23.45 ms
Queries Per Second: 80.32
================================================================================
```

### Results File

The script saves detailed results to a JSON file:

```json
{
  "training_time": 45.32,
  "model_size_bytes": 2560000,
  "indexing_time": 67.89,
  "avg_recall_at_k": 0.9234,
  "avg_precision_at_k": 0.9234,
  "avg_search_latency_ms": 12.45,
  "p50_search_latency_ms": 11.23,
  "p95_search_latency_ms": 18.67,
  "p99_search_latency_ms": 23.45,
  "qps": 80.32,
  "recalls": [...],
  "precisions": [...]
}
```

## Parameter Tuning Guide

### For Higher Accuracy
- Increase `fine_clusters` (e.g., 512, 1024)
- Increase `pq_nbits` (e.g., 12, 16)
- Increase number of training vectors
- Use more `coarse_clusters`

### For Faster Search
- Decrease `fine_clusters` (e.g., 128)
- Decrease `pq_nbits` (e.g., 6, 4)
- Decrease `coarse_clusters`

### For Lower Memory
- Decrease `pq_nbits` (e.g., 4, 6)
- Decrease `fine_clusters`
- Use smaller `pq_m` value

### Recommended Starting Points

**Small dataset (<100K vectors):**
```bash
--coarse-clusters 128 \
--fine-clusters 128 \
--pq-m 8 \
--pq-nbits 8
```

**Medium dataset (100K-1M vectors):**
```bash
--coarse-clusters 256 \
--fine-clusters 256 \
--pq-m 8 \
--pq-nbits 8
```

**Large dataset (>1M vectors):**
```bash
--coarse-clusters 512 \
--fine-clusters 256 \
--pq-m 8 \
--pq-nbits 8
```

## Comparing with FAISS IVF

To compare Puck with FAISS IVF, run the same test with FAISS:

```python
# Change method to FAISS IVF
training_request = {
    "training_index": training_index,
    "training_field": "train_vector",
    "dimension": dimension,
    "method": {
        "name": "ivf",
        "engine": "faiss",
        "space_type": "l2",
        "parameters": {
            "nlist": 256,  # Similar to coarse_clusters
            "encoder": {
                "name": "pq",
                "parameters": {
                    "m": 8,  # Similar to pq_m
                    "code_size": 8  # Similar to pq_nbits
                }
            }
        }
    }
}
```

## Troubleshooting

### Error: "Model description cannot contain any commas"
OpenSearch does not allow commas in model descriptions.

**Solution:**
```python
# ❌ Bad - contains commas
"description": "Puck model, trained on 10K vectors, with PQ"

# ✅ Good - no commas
"description": "Puck model trained on 10K vectors with PQ"
```

### Training fails with "Not enough training vectors"
Ensure you have at least 10x more training vectors than `coarse_clusters`.

**Solution:**
```bash
python test_puck_accuracy.py --num-train 5000 --coarse-clusters 256
# 5000 < 2560, increase training vectors or reduce clusters
```

### Low recall/precision
Try increasing the quality parameters:

```bash
python test_puck_accuracy.py \
    --fine-clusters 512 \
    --pq-nbits 12 \
    --num-train 20000
```

### Out of memory during training
Reduce the model complexity:

```bash
python test_puck_accuracy.py \
    --coarse-clusters 128 \
    --fine-clusters 128 \
    --pq-nbits 6
```

### Search is too slow
Reduce search complexity:

```bash
python test_puck_accuracy.py \
    --fine-clusters 128 \
    --coarse-clusters 128
```

## Advanced Usage

### Using Real Datasets

You can modify the script to use real datasets (e.g., from ANN benchmarks):

```python
# Load SIFT dataset
import h5py

with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
    train_vectors = f['train'][:]
    database_vectors = f['test'][:]
    query_vectors = f['queries'][:]
    ground_truth = f['neighbors'][:]
```

### Batch Size Tuning

For large datasets, adjust the batch size:

```python
bulk(client, actions, chunk_size=1000, refresh=False)
```

Then refresh after all batches:
```python
client.indices.refresh(index=index_name)
```

## References

- [PUCK_TRAINING_FLOW.md](../PUCK_TRAINING_FLOW.md) - Complete implementation details
- [PUCK_TRAINING_GUIDE.md](../PUCK_TRAINING_GUIDE.md) - User guide for Puck training
- [OpenSearch k-NN Documentation](https://opensearch.org/docs/latest/search-plugins/knn/)
