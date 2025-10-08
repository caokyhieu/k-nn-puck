/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.puck;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultHnswSearchContext;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;
import org.opensearch.knn.index.mapper.VectorTransformer;
import org.opensearch.knn.index.mapper.VectorTransformerFactory;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HIERARCHICAL_CLUSTER;

/**
 * Implements Puck's Hierarchical Clustering method with two-layer product quantization
 */
public class PuckHierarchicalMethod extends AbstractKNNMethod {

    // Supported vector data types
    private final static Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    // Supported space types
    private final static Set<SpaceType> SUPPORTED_SPACES = ImmutableSet.of(SpaceType.L2, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(METHOD_HIERARCHICAL_CLUSTER)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter("coarse_clusters", new Parameter.IntegerParameter("coarse_clusters", 256, (v, context) -> v > 0))
        .addParameter("fine_clusters", new Parameter.IntegerParameter("fine_clusters", 256, (v, context) -> v > 0))
        .addParameter("pq_m", new Parameter.IntegerParameter("pq_m", 8, (v, context) -> v > 0))
        .addParameter("pq_nbits", new Parameter.IntegerParameter("pq_nbits", 8, (v, context) -> v > 0 && v <= 16))
        .setRequiresTraining(true)  // Puck requires training like FAISS IVF
        .build();

    private final static KNNLibrarySearchContext SEARCH_CONTEXT = new DefaultHnswSearchContext();

    public PuckHierarchicalMethod() {
        super(METHOD_COMPONENT, SUPPORTED_SPACES, SEARCH_CONTEXT);
    }

    // Training requirement is now handled by METHOD_COMPONENT.setRequiresTraining(true)
    // The default implementation in AbstractKNNMethod will use that setting

    @Override
    protected PerDimensionValidator doGetPerDimensionValidator(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();
        if (VectorDataType.FLOAT == vectorDataType) {
            return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
        }
        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }

    @Override
    protected PerDimensionProcessor doGetPerDimensionProcessor(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        VectorDataType vectorDataType = knnMethodConfigContext.getVectorDataType();
        if (VectorDataType.FLOAT == vectorDataType) {
            return PerDimensionProcessor.NOOP_PROCESSOR;
        }
        throw new IllegalStateException("Unsupported vector data type " + vectorDataType);
    }

    @Override
    protected Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> doGetTrainingConfigValidationSetup() {
        return (trainingConfigValidationInput) -> {
            KNNMethodContext knnMethodContext = trainingConfigValidationInput.getKnnMethodContext();
            KNNMethodConfigContext knnMethodConfigContext = trainingConfigValidationInput.getKnnMethodConfigContext();
            Long trainingVectors = trainingConfigValidationInput.getTrainingVectorsCount();

            TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();

            // Validate pq_m is divisible by vector dimension
            if (knnMethodContext != null && knnMethodConfigContext != null) {
                Map<String, Object> parameters = knnMethodContext.getMethodComponentContext().getParameters();

                if (parameters.containsKey("pq_m")) {
                    int pqM = (Integer) parameters.get("pq_m");
                    int dimension = knnMethodConfigContext.getDimension();

                    if (dimension % pqM != 0) {
                        builder.valid(false);
                        return builder.build();
                    }
                }
                builder.valid(true);
            }

            // Validate number of training vectors meets minimum requirements
            // Puck hierarchical clustering needs enough vectors for both coarse and fine clusters
            if (knnMethodContext != null && trainingVectors != null) {
                Map<String, Object> parameters = knnMethodContext.getMethodComponentContext().getParameters();

                int coarseClusters = (Integer) parameters.getOrDefault("coarse_clusters", 256);
                int fineClusters = (Integer) parameters.getOrDefault("fine_clusters", 256);

                // Minimum training vectors should be at least 10x the number of coarse clusters
                // This is a conservative estimate to ensure stable clustering
                long minTrainingVectorCount = Math.max(1000, coarseClusters * 10L);

                if (trainingVectors < minTrainingVectorCount) {
                    builder.valid(false).minTrainingVectorCount(minTrainingVectorCount);
                    return builder.build();
                }
                builder.valid(true);
            }

            return builder.build();
        };
    }

    @Override
    protected VectorTransformer getVectorTransformer(SpaceType spaceType) {
        return VectorTransformerFactory.getVectorTransformer(KNNEngine.PUCK, spaceType);
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        // Puck uses compressed vectors (~1/4 original size) but needs overhead for cluster centers
        int dimension = knnMethodConfigContext.getDimension() != null ? knnMethodConfigContext.getDimension() : 128;

        // Estimate based on coarse and fine clusters
        MethodComponentContext methodComponentContext = knnMethodContext.getMethodComponentContext();
        int coarseClusters = (Integer) methodComponentContext.getParameters().getOrDefault("coarse_clusters", 256);
        int fineClusters = (Integer) methodComponentContext.getParameters().getOrDefault("fine_clusters", 256);

        // Rough estimation: cluster centers overhead
        int clusterOverhead = (coarseClusters + fineClusters) * dimension * 4 / 1024; // 4 bytes per float

        // Estimate base overhead for index structures
        int baseOverhead = 1024; // 1MB base overhead

        return clusterOverhead + baseOverhead;
    }

    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        MethodComponentContext methodComponentContext = knnMethodContext.getMethodComponentContext();

        // Build parameters map for Puck hierarchical clustering
        // Use HashMap instead of ImmutableMap to allow TrainingJob to add additional parameters
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("coarse_clusters", methodComponentContext.getParameters().getOrDefault("coarse_clusters", 256));
        parameters.put("fine_clusters", methodComponentContext.getParameters().getOrDefault("fine_clusters", 256));
        parameters.put("pq_m", methodComponentContext.getParameters().getOrDefault("pq_m", 8));
        parameters.put("pq_nbits", methodComponentContext.getParameters().getOrDefault("pq_nbits", 8));

        return KNNLibraryIndexingContextImpl.builder()
            .parameters(parameters)
            .vectorValidator(doGetVectorValidator(knnMethodContext, knnMethodConfigContext))
            .perDimensionValidator(doGetPerDimensionValidator(knnMethodContext, knnMethodConfigContext))
            .perDimensionProcessor(doGetPerDimensionProcessor(knnMethodContext, knnMethodConfigContext))
            .vectorTransformer(getVectorTransformer(knnMethodContext.getSpaceType()))
            .trainingConfigValidationSetup(doGetTrainingConfigValidationSetup())
            .build();
    }
}
