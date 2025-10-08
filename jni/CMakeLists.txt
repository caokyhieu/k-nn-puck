#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24.0)

project(KNNPlugin_JNI)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros.cmake)

# ---------------------------------- SETUP ----------------------------------
# Target libraries to be compiled
# Shared library with common utilities. Not a JNI library. Other JNI libs should depend on this one.
set(TARGET_LIB_UTIL opensearchknn_util)
set(TARGET_LIB_COMMON opensearchknn_common)  # common lib for JNI
set(TARGET_LIB_NMSLIB opensearchknn_nmslib)  # nmslib JNI
set(TARGET_LIB_FAISS opensearchknn_faiss)    # faiss JNI
set(TARGET_LIB_PUCK opensearchknn_puck)      # puck JNI
set(TARGET_LIBS "")  # Libs to be installed

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
option(CONFIG_FAISS "Configure faiss library build when this is on")
option(CONFIG_NMSLIB "Configure nmslib library build when this is on")
option(CONFIG_PUCK "Configure puck library build when this is on")
option(CONFIG_TEST "Configure tests when this is on")

if (${CONFIG_FAISS} STREQUAL OFF AND ${CONFIG_NMSLIB} STREQUAL OFF AND ${CONFIG_PUCK} STREQUAL OFF AND ${CONFIG_TEST} STREQUAL OFF)
    set(CONFIG_ALL ON)
else()
    set(CONFIG_ALL OFF)
endif ()

# `git am` will create commits from the patches in the native libraries. This is ideal for development envs
# because it prevents full lib rebuild everytime cmake is run. However, for build systems that will run the
# build workflow once, it can cause issues because git commits require that the user and the user's email be set.
# See https://github.com/opensearch-project/k-NN/issues/1651. So, we provide a flag that allows users to select between
# the two
if(NOT DEFINED COMMIT_LIB_PATCHES OR "${COMMIT_LIB_PATCHES}" STREQUAL true)
    set(GIT_PATCH_COMMAND am)
else()
    set(GIT_PATCH_COMMAND apply)
endif()
message(STATUS "Using the following git patch command: \"${GIT_PATCH_COMMAND}\"")

# Set OS specific variables
if (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    set(CMAKE_MACOSX_RPATH 1)
    set(JVM_OS_TYPE darwin)
    set(LIB_EXT .jnilib)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    set(JVM_OS_TYPE linux)
    set(LIB_EXT .so)
    # Set RPATH to $ORIGIN so libraries can find each other in the same directory
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
# Set the CXX_COMPILER_VERSION, CMAKE_CXX_FLAGS, JVM_OS_TYPE, prefix and extension for the target libraries that are built.
    set(CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
    set(JVM_OS_TYPE win32)
    set(LIB_EXT .dll)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
else()
    message(FATAL_ERROR "Unable to run on system: ${CMAKE_SYSTEM_NAME}")
endif()

# By default, set to 0.0.0
if(NOT KNN_PLUGIN_VERSION)
    set(KNN_PLUGIN_VERSION "0.0.0")
endif()

# Set architecture specific variables
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64)
    set(MACH_ARCH arm64)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
    set(MACH_ARCH x64)
endif()
# ----------------------------------------------------------------------------

# ---------------------------------- UTIL ----------------------------------
add_library(${TARGET_LIB_UTIL} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/jni_util.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/commons.cpp)
target_include_directories(${TARGET_LIB_UTIL} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include $ENV{JAVA_HOME}/include $ENV{JAVA_HOME}/include/${JVM_OS_TYPE})
opensearch_set_common_properties(${TARGET_LIB_UTIL})
list(APPEND TARGET_LIBS ${TARGET_LIB_UTIL})
# ----------------------------------------------------------------------------

# ---------------------------------- COMMON ----------------------------------
add_library(${TARGET_LIB_COMMON} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/org_opensearch_knn_jni_JNICommons.cpp)
target_link_libraries(${TARGET_LIB_COMMON} ${TARGET_LIB_UTIL})
target_include_directories(${TARGET_LIB_COMMON} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include $ENV{JAVA_HOME}/include $ENV{JAVA_HOME}/include/${JVM_OS_TYPE})
opensearch_set_common_properties(${TARGET_LIB_COMMON})
list(APPEND TARGET_LIBS ${TARGET_LIB_COMMON})
# ----------------------------------------------------------------------------

# ---------------------------------- NMSLIB ----------------------------------
if (${CONFIG_NMSLIB} STREQUAL ON OR ${CONFIG_ALL} STREQUAL ON OR ${CONFIG_TEST} STREQUAL ON)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/init-nmslib.cmake)
    add_library(${TARGET_LIB_NMSLIB} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/org_opensearch_knn_jni_NmslibService.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/nmslib_wrapper.cpp)
    target_link_libraries(${TARGET_LIB_NMSLIB} NonMetricSpaceLib ${TARGET_LIB_UTIL})
    target_include_directories(${TARGET_LIB_NMSLIB} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include $ENV{JAVA_HOME}/include $ENV{JAVA_HOME}/include/${JVM_OS_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib/similarity_search/include)
    opensearch_set_common_properties(${TARGET_LIB_NMSLIB})
    list(APPEND TARGET_LIBS ${TARGET_LIB_NMSLIB})
endif ()

# ---------------------------------------------------------------------------

# ---------------------------------- FAISS ----------------------------------
if (${CONFIG_FAISS} STREQUAL ON OR ${CONFIG_ALL} STREQUAL ON OR ${CONFIG_TEST} STREQUAL ON)
    # Force use of shared libraries for BLAS/LAPACK to avoid OpenMP static linking issues
    set(BLA_STATIC OFF)

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/init-faiss.cmake)
    find_package(OpenMP REQUIRED)

    # Force use of shared OpenMP library instead of static to avoid relocation errors
    set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -fopenmp")
    add_library(
        ${TARGET_LIB_FAISS} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/org_opensearch_knn_jni_FaissService.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/faiss_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/faiss_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/faiss_index_service.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/faiss_methods.cpp
    )
    # Explicitly link shared gomp library to prevent static linking
    target_link_libraries(${TARGET_LIB_FAISS} ${TARGET_LINK_FAISS_LIB} ${TARGET_LIB_UTIL} OpenMP::OpenMP_CXX -lgomp)
    target_include_directories(${TARGET_LIB_FAISS} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $ENV{JAVA_HOME}/include
        $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss
    )
    opensearch_set_common_properties(${TARGET_LIB_FAISS})
    list(APPEND TARGET_LIBS ${TARGET_LIB_FAISS})
endif ()

# ---------------------------------- PUCK ----------------------------------
if (${CONFIG_PUCK} STREQUAL ON OR ${CONFIG_ALL} STREQUAL ON)
    # Initialize Puck submodule and apply patches
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/init-puck.cmake)

    # Define FINTEGER for MKL compatibility
    add_definitions(-DFINTEGER=int)

    # Add Puck cmake modules to path for FindMKL
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/puck/cmake")

    # Find MKL package
    find_package(MKL REQUIRED)

    # Fix MKL_H variable - FindMKL.cmake sets it incorrectly
    # MKL_H should point to the directory containing mkl.h, not the file itself
    # This must be done BEFORE add_subdirectory to affect Puck's CMakeLists
    if(DEFINED ENV{MKLROOT})
        set(MKL_H "$ENV{MKLROOT}/include" CACHE PATH "MKL include directory" FORCE)
        set(MKL_INCLUDE_DIR "$ENV{MKLROOT}/include" CACHE PATH "MKL include directory" FORCE)
    endif()

    # Add Puck subdirectory and configure build
    add_subdirectory(external/puck)

    # Create Puck JNI library
    add_library(
        ${TARGET_LIB_PUCK} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/org_opensearch_knn_jni_PuckService.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/puck_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/puck_index_service.cpp
    )

    # Link with Puck library and utilities
    # Add shared gomp library explicitly to avoid static linking issues
    target_link_libraries(${TARGET_LIB_PUCK} puck ${TARGET_LIB_UTIL} ${MKL_LIBRARIES} -lgomp)

    target_include_directories(${TARGET_LIB_PUCK} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $ENV{JAVA_HOME}/include
        $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/puck
        ${CMAKE_CURRENT_SOURCE_DIR}/external/puck/puck
        ${GFLAGS_INCLUDE_DIR}
        ${GLOG_INCLUDE_DIR}
        ${MKL_INCLUDE_DIR}
    )

    opensearch_set_common_properties(${TARGET_LIB_PUCK})
    list(APPEND TARGET_LIBS ${TARGET_LIB_PUCK})
endif ()

# ---------------------------------------------------------------------------

# --------------------------------- TESTS -----------------------------------
# Windows : Comment the TESTS for now because the tests are failing(failing to build jni_tests.exe) if we are building our target libraries as SHARED libraries.
# TODO: Fix the failing JNI TESTS on Windows
if ("${WIN32}" STREQUAL "")
    if (${CONFIG_ALL} STREQUAL ON OR ${CONFIG_TEST} STREQUAL ON)
        # Reference - https://crascit.com/2015/07/25/cmake-gtest/
        configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
        execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
                )
        execute_process(COMMAND "${CMAKE_COMMAND}" --build .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
                )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                "${CMAKE_BINARY_DIR}/googletest-build" EXCLUDE_FROM_ALL
                )
        add_executable(
                jni_test
                tests/faiss_wrapper_test.cpp
                tests/faiss_wrapper_unit_test.cpp
                tests/faiss_util_test.cpp
                tests/nmslib_wrapper_test.cpp
                tests/nmslib_wrapper_unit_test.cpp
                tests/test_util.cpp
                tests/commons_test.cpp
                tests/faiss_stream_support_test.cpp
                tests/faiss_index_service_test.cpp
                tests/nmslib_stream_support_test.cpp
        )

        target_link_libraries(
                jni_test
                gtest_main
                gmock_main
                faiss
                NonMetricSpaceLib
                OpenMP::OpenMP_CXX
                ${TARGET_LIB_FAISS}
                ${TARGET_LIB_NMSLIB}
                ${TARGET_LIB_COMMON}
                ${TARGET_LIB_UTIL}
        )

        target_include_directories(jni_test PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/tests
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                $ENV{JAVA_HOME}/include
                $ENV{JAVA_HOME}/include/${JVM_OS_TYPE}
                ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss
                ${CMAKE_CURRENT_SOURCE_DIR}/external/nmslib/similarity_search/include
                ${gtest_SOURCE_DIR}/include
                ${gmock_SOURCE_DIR}/include)


        set_target_properties(jni_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
    endif ()
endif()

# ---------------------------------------------------------------------------
