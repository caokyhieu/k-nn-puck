/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.puck;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;

import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HIERARCHICAL_CLUSTER;

/**
 * Implements NativeLibrary for the Puck native library
 */
public class Puck extends NativeLibrary {
    public final static String EXTENSION = ".puck";
    final static String CURRENT_VERSION = "1.0.0";

    // Puck supports hierarchical clustering method
    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(METHOD_HIERARCHICAL_CLUSTER, new PuckHierarchicalMethod());

    // Score translation for supported space types
    private final static Map<SpaceType, Function<Float, Float>> SCORE_TRANSLATIONS = ImmutableMap.of(
        SpaceType.L2,
        rawScore -> rawScore,
        SpaceType.COSINESIMIL,
        rawScore -> 1.0f - rawScore,
        SpaceType.INNER_PRODUCT,
        rawScore -> -rawScore
    );

    public final static Puck INSTANCE = new Puck(METHODS, SCORE_TRANSLATIONS, CURRENT_VERSION, EXTENSION);
    private final MethodResolver methodResolver;

    /**
     * Constructor for Puck
     *
     * @param methods Set of methods the native library supports
     * @param scoreTranslation Map of translation of space type to scores returned by the library
     * @param currentVersion String representation of current version of the library
     * @param extension String representing the extension that library files should use
     */
    private Puck(
        Map<String, KNNMethod> methods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String currentVersion,
        String extension
    ) {
        super(methods, scoreTranslation, currentVersion, extension);
        this.methodResolver = new PuckMethodResolver();
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        // For Puck, distance is used as is - no transformation needed
        return distance;
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        // For Puck, convert score to distance using space type translation
        return spaceType.scoreToDistanceTranslation(score);
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }
}
