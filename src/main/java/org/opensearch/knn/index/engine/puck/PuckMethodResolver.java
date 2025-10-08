/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.puck;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import static org.opensearch.knn.common.KNNConstants.METHOD_HIERARCHICAL_CLUSTER;

/**
 * Method resolver for Puck library
 */
public class PuckMethodResolver extends AbstractMethodResolver {

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        String methodName = knnMethodContext.getMethodComponentContext().getName();

        // For now, Puck only supports hierarchical_cluster method
        if (METHOD_HIERARCHICAL_CLUSTER.equals(methodName)) {
            return ResolvedMethodContext.builder().knnMethodContext(knnMethodContext).compressionLevel(CompressionLevel.x4).build();
        }

        throw new IllegalArgumentException("Unsupported method: " + methodName + " for Puck engine");
    }
}
