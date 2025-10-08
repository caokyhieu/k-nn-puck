#!/usr/bin/env python3
"""
Puck Index Testing Script - Accuracy and Performance Evaluation

This script tests the Puck hierarchical cluster index implementation in OpenSearch,
measuring recall@K, precision@K, training time, indexing time, and search latency.

Requirements:
    pip install opensearch-py numpy scikit-learn h5py

Usage:
    python test_puck_accuracy.py --host localhost --port 9200
"""

import argparse
import time
import numpy as np
from typing import List, Tuple, Dict
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import json


class PuckAccuracyTester:
    def __init__(self, host: str = 'localhost', port: int = 9200, use_ssl: bool = False):
        """Initialize OpenSearch client"""
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )

    def generate_random_vectors(self, num_vectors: int, dimension: int) -> np.ndarray:
        """Generate random normalized vectors for testing"""
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        return vectors

    def create_training_index(self, index_name: str, dimension: int) -> None:
        """Create index for training vectors"""
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        mapping = {
            "mappings": {
                "properties": {
                    "train_vector": {
                        "type": "knn_vector",
                        "dimension": dimension
                    }
                }
            }
        }

        self.client.indices.create(index=index_name, body=mapping)
        print(f"✓ Created training index: {index_name}")

    def index_training_vectors(self, index_name: str, vectors: np.ndarray) -> None:
        """Index training vectors using bulk API"""
        actions = []
        for i, vector in enumerate(vectors):
            actions.append({
                "_index": index_name,
                "_id": str(i),
                "_source": {
                    "train_vector": vector.tolist()
                }
            })

        success, failed = bulk(self.client, actions, chunk_size=500, refresh=True)
        print(f"✓ Indexed {success} training vectors")

        # Refresh to make sure all docs are searchable
        self.client.indices.refresh(index=index_name)

    def train_model(
        self,
        model_id: str,
        training_index: str,
        dimension: int,
        coarse_clusters: int = 256,
        fine_clusters: int = 256,
        pq_m: int = 8,
        pq_nbits: int = 8
    ) -> Dict:
        """Train a Puck model"""
        training_request = {
            "training_index": training_index,
            "training_field": "train_vector",
            "dimension": dimension,
            "description": "Puck test model for accuracy evaluation",
            "method": {
                "name": "hierarchical_cluster",
                "engine": "puck",
                "space_type": "l2",
                "parameters": {
                    "coarse_clusters": coarse_clusters,
                    "fine_clusters": fine_clusters,
                    "pq_m": pq_m,
                    "pq_nbits": pq_nbits
                }
            }
        }

        start_time = time.time()

        # Delete model if it exists
        try:
            self.client.transport.perform_request(
                'DELETE',
                f'/_plugins/_knn/models/{model_id}'
            )
            time.sleep(1)
        except:
            pass

        # Start training
        response = self.client.transport.perform_request(
            'POST',
            f'/_plugins/_knn/models/{model_id}/_train',
            body=training_request
        )

        print(f"✓ Training started for model: {model_id}")
        print(f"  State: {response.get('state', 'unknown')}")

        # Wait for training to complete
        while True:
            model_info = self.client.transport.perform_request(
                'GET',
                f'/_plugins/_knn/models/{model_id}'
            )

            state = model_info.get('state', 'unknown')
            print(f"  Training state: {state}", end='\r')

            if state == 'created':
                training_time = time.time() - start_time
                print(f"\n✓ Training completed in {training_time:.2f} seconds")
                return {
                    'training_time': training_time,
                    'model_size': model_info.get('model_blob_size_in_bytes', 0)
                }
            elif state == 'failed':
                error = model_info.get('error', 'Unknown error')
                raise Exception(f"Training failed: {error}")

            time.sleep(2)

    def create_index_with_model(
        self,
        index_name: str,
        model_id: str,
        dimension: int
    ) -> None:
        """Create an index using a trained model"""
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        mapping = {
            "settings": {
                "index.knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "model_id": model_id
                    },
                    "doc_id": {
                        "type": "keyword"
                    }
                }
            }
        }

        self.client.indices.create(index=index_name, body=mapping)
        print(f"✓ Created index with trained model: {index_name}")

    def index_vectors(self, index_name: str, vectors: np.ndarray) -> float:
        """Index vectors and return indexing time"""
        start_time = time.time()

        actions = []
        for i, vector in enumerate(vectors):
            actions.append({
                "_index": index_name,
                "_id": str(i),
                "_source": {
                    "vector": vector.tolist(),
                    "doc_id": str(i)
                }
            })

        success, failed = bulk(self.client, actions, chunk_size=500, refresh=True)

        indexing_time = time.time() - start_time
        print(f"✓ Indexed {success} vectors in {indexing_time:.2f} seconds")
        print(f"  Throughput: {success / indexing_time:.2f} docs/sec")

        # Force merge to optimize segments
        self.client.indices.forcemerge(index=index_name, max_num_segments=1)

        return indexing_time

    def compute_ground_truth(
        self,
        query_vectors: np.ndarray,
        database_vectors: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Compute ground truth nearest neighbors using brute force L2 distance"""
        num_queries = query_vectors.shape[0]
        ground_truth = np.zeros((num_queries, k), dtype=np.int32)

        print(f"Computing ground truth for {num_queries} queries...")
        for i, query in enumerate(query_vectors):
            # Compute L2 distances
            distances = np.sum((database_vectors - query) ** 2, axis=1)
            # Get top-k nearest neighbors
            nearest = np.argsort(distances)[:k]
            ground_truth[i] = nearest

        return ground_truth

    def search_knn(
        self,
        index_name: str,
        query_vector: np.ndarray,
        k: int
    ) -> Tuple[List[int], float]:
        """Search for k nearest neighbors and return IDs and search time"""
        start_time = time.time()

        query = {
            "size": k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector.tolist(),
                        "k": k
                    }
                }
            }
        }

        response = self.client.search(index=index_name, body=query)

        search_time = time.time() - start_time

        # Extract document IDs
        result_ids = [int(hit['_source']['doc_id']) for hit in response['hits']['hits']]

        return result_ids, search_time

    def calculate_recall_at_k(
        self,
        predicted: List[int],
        ground_truth: List[int],
        k: int
    ) -> float:
        """Calculate recall@k"""
        predicted_set = set(predicted[:k])
        ground_truth_set = set(ground_truth[:k])

        if len(ground_truth_set) == 0:
            return 0.0

        intersection = predicted_set.intersection(ground_truth_set)
        return len(intersection) / len(ground_truth_set)

    def calculate_precision_at_k(
        self,
        predicted: List[int],
        ground_truth: List[int],
        k: int
    ) -> float:
        """Calculate precision@k"""
        predicted_set = set(predicted[:k])
        ground_truth_set = set(ground_truth[:k])

        if len(predicted_set) == 0:
            return 0.0

        intersection = predicted_set.intersection(ground_truth_set)
        return len(intersection) / len(predicted_set)

    def run_accuracy_test(
        self,
        num_train: int = 10000,
        num_database: int = 50000,
        num_queries: int = 100,
        dimension: int = 128,
        k: int = 10,
        coarse_clusters: int = 256,
        fine_clusters: int = 256,
        pq_m: int = 8,
        pq_nbits: int = 8
    ) -> Dict:
        """Run complete accuracy and performance test"""
        print("\n" + "="*80)
        print("PUCK ACCURACY AND PERFORMANCE TEST")
        print("="*80)
        print(f"Configuration:")
        print(f"  Training vectors: {num_train}")
        print(f"  Database vectors: {num_database}")
        print(f"  Query vectors: {num_queries}")
        print(f"  Dimension: {dimension}")
        print(f"  K: {k}")
        print(f"  Coarse clusters: {coarse_clusters}")
        print(f"  Fine clusters: {fine_clusters}")
        print(f"  PQ subspaces (m): {pq_m}")
        print(f"  PQ bits: {pq_nbits}")
        print("="*80 + "\n")

        # Step 1: Generate vectors
        print("Step 1: Generating vectors...")
        train_vectors = self.generate_random_vectors(num_train, dimension)
        database_vectors = self.generate_random_vectors(num_database, dimension)
        query_vectors = self.generate_random_vectors(num_queries, dimension)
        print(f"✓ Generated {num_train} training, {num_database} database, {num_queries} query vectors\n")

        # Step 2: Create and populate training index
        print("Step 2: Creating training index...")
        training_index = "puck_train_index"
        self.create_training_index(training_index, dimension)
        self.index_training_vectors(training_index, train_vectors)
        print()

        # Step 3: Train model
        print("Step 3: Training Puck model...")
        model_id = "puck_test_model"
        training_stats = self.train_model(
            model_id,
            training_index,
            dimension,
            coarse_clusters,
            fine_clusters,
            pq_m,
            pq_nbits
        )
        print()

        # Step 4: Create index with trained model
        print("Step 4: Creating index with trained model...")
        test_index = "puck_test_index"
        self.create_index_with_model(test_index, model_id, dimension)
        print()

        # Step 5: Index database vectors
        print("Step 5: Indexing database vectors...")
        indexing_time = self.index_vectors(test_index, database_vectors)
        print()

        # Step 6: Compute ground truth
        print("Step 6: Computing ground truth...")
        ground_truth = self.compute_ground_truth(query_vectors, database_vectors, k)
        print("✓ Ground truth computed\n")

        # Step 7: Run queries and measure accuracy
        print("Step 7: Running queries and measuring accuracy...")
        recalls = []
        precisions = []
        search_times = []

        for i, query_vec in enumerate(query_vectors):
            result_ids, search_time = self.search_knn(test_index, query_vec, k)

            recall = self.calculate_recall_at_k(result_ids, ground_truth[i].tolist(), k)
            precision = self.calculate_precision_at_k(result_ids, ground_truth[i].tolist(), k)

            recalls.append(recall)
            precisions.append(precision)
            search_times.append(search_time)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_queries} queries...", end='\r')

        print(f"\n✓ Completed {num_queries} queries\n")

        # Calculate statistics
        results = {
            'training_time': training_stats['training_time'],
            'model_size_bytes': training_stats['model_size'],
            'indexing_time': indexing_time,
            'avg_recall_at_k': np.mean(recalls),
            'avg_precision_at_k': np.mean(precisions),
            'avg_search_latency_ms': np.mean(search_times) * 1000,
            'p50_search_latency_ms': np.percentile(search_times, 50) * 1000,
            'p95_search_latency_ms': np.percentile(search_times, 95) * 1000,
            'p99_search_latency_ms': np.percentile(search_times, 99) * 1000,
            'qps': 1.0 / np.mean(search_times),
            'recalls': recalls,
            'precisions': precisions
        }

        # Print results
        print("="*80)
        print("RESULTS")
        print("="*80)
        print(f"Training Time: {results['training_time']:.2f} seconds")
        print(f"Model Size: {results['model_size_bytes'] / (1024*1024):.2f} MB")
        print(f"Indexing Time: {results['indexing_time']:.2f} seconds")
        print(f"Indexing Throughput: {num_database / results['indexing_time']:.2f} docs/sec")
        print()
        print(f"Average Recall@{k}: {results['avg_recall_at_k']:.4f} ({results['avg_recall_at_k']*100:.2f}%)")
        print(f"Average Precision@{k}: {results['avg_precision_at_k']:.4f} ({results['avg_precision_at_k']*100:.2f}%)")
        print()
        print(f"Average Search Latency: {results['avg_search_latency_ms']:.2f} ms")
        print(f"P50 Search Latency: {results['p50_search_latency_ms']:.2f} ms")
        print(f"P95 Search Latency: {results['p95_search_latency_ms']:.2f} ms")
        print(f"P99 Search Latency: {results['p99_search_latency_ms']:.2f} ms")
        print(f"Queries Per Second: {results['qps']:.2f}")
        print("="*80 + "\n")

        # Cleanup
        print("Cleaning up...")
        try:
            self.client.indices.delete(index=training_index)
            self.client.indices.delete(index=test_index)
            self.client.transport.perform_request('DELETE', f'/_plugins/_knn/models/{model_id}')
            print("✓ Cleanup completed\n")
        except:
            pass

        return results


def main():
    parser = argparse.ArgumentParser(description='Test Puck index accuracy and performance')
    parser.add_argument('--host', type=str, default='localhost', help='OpenSearch host')
    parser.add_argument('--port', type=int, default=9200, help='OpenSearch port')
    parser.add_argument('--num-train', type=int, default=10000, help='Number of training vectors')
    parser.add_argument('--num-database', type=int, default=50000, help='Number of database vectors')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of query vectors')
    parser.add_argument('--dimension', type=int, default=128, help='Vector dimension')
    parser.add_argument('--k', type=int, default=10, help='K for top-K search')
    parser.add_argument('--coarse-clusters', type=int, default=256, help='Number of coarse clusters')
    parser.add_argument('--fine-clusters', type=int, default=256, help='Number of fine clusters')
    parser.add_argument('--pq-m', type=int, default=8, help='PQ subspaces')
    parser.add_argument('--pq-nbits', type=int, default=8, help='PQ bits')
    parser.add_argument('--use-ssl', action='store_true', help='Use SSL connection')

    args = parser.parse_args()

    # Validate parameters
    if args.dimension % args.pq_m != 0:
        print(f"Error: dimension ({args.dimension}) must be divisible by pq_m ({args.pq_m})")
        return

    # Run test
    tester = PuckAccuracyTester(args.host, args.port, args.use_ssl)

    results = tester.run_accuracy_test(
        num_train=args.num_train,
        num_database=args.num_database,
        num_queries=args.num_queries,
        dimension=args.dimension,
        k=args.k,
        coarse_clusters=args.coarse_clusters,
        fine_clusters=args.fine_clusters,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits
    )

    # Save results to JSON
    output_file = f"puck_test_results_d{args.dimension}_k{args.k}.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        results_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else
               [float(x) for x in v] if isinstance(v, list) else v
            for k, v in results.items()
        }
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
