/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

/**
 * Service to distribute Puck requests to JNI layer
 */
public class PuckService {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            System.loadLibrary(KNNConstants.PUCK_JNI_LIBRARY_NAME);
            initLibrary();
            KNNEngine.PUCK.setInitialized(true);
            return null;
        });
    }

    /**
     * Initialize the Puck library
     */
    private static native void initLibrary();

    /**
     * Train a Puck index using training vectors
     *
     * @param indexParameters parameters for index training (coarse_clusters, fine_clusters, pq_m, pq_nbits)
     * @param dimension       dimension of vectors
     * @param trainVectorsPointer pointer to training vectors in native memory
     * @return byte array of trained index
     */
    public static native byte[] trainIndex(Map<String, Object> indexParameters, int dimension, long trainVectorsPointer);

    /**
     * Create an index for the Puck library.
     *
     * @param ids            array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param output         Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param parameters     parameters to build index
     */
    public static native void createIndex(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        Map<String, Object> parameters
    );

    /**
     * Initialize an index for the Puck library.
     *
     * @param numDocs    number of documents to be added
     * @param dim        dimension of the vector to be indexed
     * @param parameters parameters to build index
     * @return address of the index in memory
     */
    public static native long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    // Note: Training is now done through trainIndex(Map, int, long) which matches FAISS API
    // This method is kept for backwards compatibility but may be deprecated
    // public static native void trainIndex(long indexPointer, float[] vectors, int dim, Map<String, Object> parameters);

    /**
     * Query the Puck index
     *
     * @param indexPointer pointer to index in memory
     * @param queryVector  query vector
     * @param k           number of nearest neighbors to return
     * @param methodParameters method parameter
     * @param parentIds list of parent doc ids when the knn field is a nested field
     * @return KNNQueryResult array
     */
    public static native KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        int[] parentIds
    );

    /**
     * Load an index into memory via a wrapping having Lucene's IndexInput.
     * Instead of directly accessing an index path, this will make Faiss delegate IndexInput to load bytes.
     *
     * @param readStream IndexInput wrapper having a Lucene's IndexInput reference.
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndexWithStream(IndexInputWithBuffer readStream);

    /**
     * Load a binary index into memory
     *
     * @param indexPath path to index file
     * @return pointer to location in memory the index resides in
     */
    public static native long loadBinaryIndex(String indexPath);

    /**
     * Load a binary index into memory with a wrapping having Lucene's IndexInput.
     * Instead of directly accessing an index path, this will make Faiss delegate IndexInput to load bytes.
     *
     * @param readStream IndexInput wrapper having a Lucene's IndexInput reference.
     * @return pointer to location in memory the index resides in
     */
    public static native long loadBinaryIndexWithStream(IndexInputWithBuffer readStream);

    /**
     * Load an index into memory
     *
     * @param indexPath path to index file
     * @return pointer to location in memory the index resides in
     */
    public static native long loadIndex(String indexPath);

    /**
     * Free the index from memory
     *
     * @param indexPointer pointer to index in memory
     */
    public static native void freeIndex(long indexPointer);

    /**
     * Create an index from a trained template (trained model blob)
     *
     * @param ids            array of ids mapping to the data passed in
     * @param vectorsAddress address of native memory where vectors are stored
     * @param dim            dimension of the vector to be indexed
     * @param output         Index output wrapper having Lucene's IndexOutput to be used to flush bytes in native engines.
     * @param templateIndex  Trained model blob (from trainIndex)
     * @param parameters     parameters to build index
     */
    public static native void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    );
}
