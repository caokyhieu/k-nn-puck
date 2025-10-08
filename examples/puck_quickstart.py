#!/usr/bin/env python3
"""
Puck Quick Start Example

A simple example showing how to:
1. Train a Puck model
2. Create an index with the trained model
3. Index vectors
4. Search for nearest neighbors

Requirements:
    pip install opensearch-py numpy
"""

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import numpy as np
import time


def main():
    # Connect to OpenSearch
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_compress=True,
        use_ssl=False
    )

    # Configuration
    DIMENSION = 128
    NUM_TRAIN_VECTORS = 10000
    NUM_INDEX_VECTORS = 50000

    print("="*80)
    print("Puck Hierarchical Cluster Index - Quick Start Example")
    print("="*80)
    print(f"Vector Dimension: {DIMENSION}")
    print(f"Training Vectors: {NUM_TRAIN_VECTORS}")
    print(f"Index Vectors: {NUM_INDEX_VECTORS}")
    print("="*80 + "\n")

    # ========================================================================
    # STEP 1: Create training index and add training vectors
    # ========================================================================
    print("Step 1: Creating training index...")

    training_index = "puck-training-example"

    # Delete if exists
    if client.indices.exists(index=training_index):
        client.indices.delete(index=training_index)

    # Create training index
    client.indices.create(
        index=training_index,
        body={
            "mappings": {
                "properties": {
                    "train_vector": {
                        "type": "knn_vector",
                        "dimension": DIMENSION
                    }
                }
            }
        }
    )

    # Generate random training vectors
    train_vectors = np.random.randn(NUM_TRAIN_VECTORS, DIMENSION).astype(np.float32)

    # Bulk index training vectors
    actions = [
        {
            "_index": training_index,
            "_id": str(i),
            "_source": {"train_vector": vec.tolist()}
        }
        for i, vec in enumerate(train_vectors)
    ]

    bulk(client, actions, chunk_size=500, refresh=True)
    print(f"✓ Indexed {NUM_TRAIN_VECTORS} training vectors\n")

    # ========================================================================
    # STEP 2: Train Puck model
    # ========================================================================
    print("Step 2: Training Puck model...")

    model_id = "my-puck-model"

    # Delete model if exists
    try:
        client.transport.perform_request('DELETE', f'/_plugins/_knn/models/{model_id}')
        time.sleep(1)
    except:
        pass

    # Train the model
    training_request = {
        "training_index": training_index,
        "training_field": "train_vector",
        "dimension": DIMENSION,
        "description": "Example Puck model for quick start",
        "method": {
            "name": "hierarchical_cluster",
            "engine": "puck",
            "space_type": "l2",
            "parameters": {
                "coarse_clusters": 256,  # sqrt(10000) ≈ 100, use 256 for better accuracy
                "fine_clusters": 256,     # Balance between accuracy and speed
                "pq_m": 8,                # 128 / 8 = 16 dimensional subspaces
                "pq_nbits": 8             # 256 centers per subspace
            }
        }
    }

    response = client.transport.perform_request(
        'POST',
        f'/_plugins/_knn/models/{model_id}/_train',
        body=training_request
    )

    print(f"Training started: {response}")

    # Wait for training to complete
    while True:
        model_info = client.transport.perform_request(
            'GET',
            f'/_plugins/_knn/models/{model_id}'
        )

        state = model_info.get('state')
        print(f"Training state: {state}", end='\r')

        if state == 'created':
            model_size_mb = model_info.get('model_blob_size_in_bytes', 0) / (1024 * 1024)
            print(f"\n✓ Training completed! Model size: {model_size_mb:.2f} MB\n")
            break
        elif state == 'failed':
            print(f"\n✗ Training failed: {model_info.get('error')}")
            return

        time.sleep(2)

    # ========================================================================
    # STEP 3: Create index with trained model
    # ========================================================================
    print("Step 3: Creating index with trained model...")

    index_name = "puck-vectors-example"

    # Delete if exists
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    # Create index with model_id
    client.indices.create(
        index=index_name,
        body={
            "settings": {
                "index.knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "model_id": model_id  # Use trained model
                    },
                    "title": {
                        "type": "text"
                    }
                }
            }
        }
    )

    print(f"✓ Created index: {index_name}\n")

    # ========================================================================
    # STEP 4: Index vectors
    # ========================================================================
    print(f"Step 4: Indexing {NUM_INDEX_VECTORS} vectors...")

    # Generate random vectors for the index
    index_vectors = np.random.randn(NUM_INDEX_VECTORS, DIMENSION).astype(np.float32)

    # Bulk index vectors
    actions = [
        {
            "_index": index_name,
            "_id": str(i),
            "_source": {
                "vector": vec.tolist(),
                "title": f"Document {i}"
            }
        }
        for i, vec in enumerate(index_vectors)
    ]

    start_time = time.time()
    success, failed = bulk(client, actions, chunk_size=500, refresh=True)
    indexing_time = time.time() - start_time

    print(f"✓ Indexed {success} vectors in {indexing_time:.2f} seconds")
    print(f"  Throughput: {success / indexing_time:.2f} docs/sec\n")

    # Force merge
    client.indices.forcemerge(index=index_name, max_num_segments=1)

    # ========================================================================
    # STEP 5: Search for nearest neighbors
    # ========================================================================
    print("Step 5: Searching for nearest neighbors...")

    # Generate a random query vector
    query_vector = np.random.randn(DIMENSION).astype(np.float32)

    # Search for top 10 nearest neighbors
    search_query = {
        "size": 10,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector.tolist(),
                    "k": 10
                }
            }
        }
    }

    start_time = time.time()
    response = client.search(index=index_name, body=search_query)
    search_time = (time.time() - start_time) * 1000  # Convert to ms

    print(f"✓ Search completed in {search_time:.2f} ms")
    print(f"\nTop 10 Results:")
    for i, hit in enumerate(response['hits']['hits'], 1):
        print(f"  {i}. Document: {hit['_source']['title']}, Score: {hit['_score']:.4f}")

    # ========================================================================
    # STEP 6: Performance test with multiple queries
    # ========================================================================
    print("\nStep 6: Running performance test (100 queries)...")

    search_times = []
    for i in range(100):
        query_vec = np.random.randn(DIMENSION).astype(np.float32)

        search_query = {
            "size": 10,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vec.tolist(),
                        "k": 10
                    }
                }
            }
        }

        start = time.time()
        client.search(index=index_name, body=search_query)
        search_times.append(time.time() - start)

    avg_latency = np.mean(search_times) * 1000
    p95_latency = np.percentile(search_times, 95) * 1000
    p99_latency = np.percentile(search_times, 99) * 1000
    qps = 1.0 / np.mean(search_times)

    print(f"✓ Performance test completed")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  P95 latency: {p95_latency:.2f} ms")
    print(f"  P99 latency: {p99_latency:.2f} ms")
    print(f"  QPS: {qps:.2f}")

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
