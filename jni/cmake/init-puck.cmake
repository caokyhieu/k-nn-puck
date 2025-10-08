#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

# Check if puck exists
find_path(PUCK_REPO_DIR NAMES puck PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/puck NO_DEFAULT_PATH)

# If not, pull the updated submodule
if (NOT EXISTS ${PUCK_REPO_DIR})
    message(STATUS "Could not find puck. Pulling updated submodule.")
    execute_process(COMMAND git submodule update --init -- external/puck WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

# Apply patches
if(NOT DEFINED APPLY_LIB_PATCHES OR "${APPLY_LIB_PATCHES}" STREQUAL true)
    # Define list of patch files
    set(PATCH_FILE_LIST)
    list(APPEND PATCH_FILE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/patches/puck/0001-fix-mkl-include-directories.patch")

    # Get patch id of the last commit
    execute_process(COMMAND sh -c "git --no-pager show HEAD | git patch-id --stable" OUTPUT_VARIABLE PATCH_ID_OUTPUT_FROM_COMMIT WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/puck)
    string(REPLACE " " ";" PATCH_ID_LIST_FROM_COMMIT ${PATCH_ID_OUTPUT_FROM_COMMIT})
    list(GET PATCH_ID_LIST_FROM_COMMIT 0 PATCH_ID_FROM_COMMIT)

    # Find all patch files need to apply
    list(SORT PATCH_FILE_LIST ORDER DESCENDING)
    foreach(PATCH_FILE ${PATCH_FILE_LIST})
        execute_process(COMMAND sh -c "git patch-id --stable < ${PATCH_FILE}" OUTPUT_VARIABLE PATCH_ID_OUTPUT_FROM_FILE)
        string(REPLACE " " ";" PATCH_ID_LIST_FROM_FILE ${PATCH_ID_OUTPUT_FROM_FILE})
        list(GET PATCH_ID_LIST_FROM_FILE 0 PATCH_ID_FROM_FILE)

        if(NOT "${PATCH_ID_FROM_COMMIT}" STREQUAL "${PATCH_ID_FROM_FILE}")
            message(STATUS "Applying patch ${PATCH_FILE}")
            execute_process(COMMAND git apply --ignore-space-change --ignore-whitespace --3way ${PATCH_FILE} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/puck RESULT_VARIABLE APPLY_STATUS)
            if(NOT APPLY_STATUS EQUAL "0")
                message(FATAL_ERROR "Failed to apply patch ${PATCH_FILE}")
            endif()
            if("${COMMIT_LIB_PATCHES}" STREQUAL true)
                execute_process(COMMAND git commit -am "Applied patch ${PATCH_FILE}" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/puck)
            endif()
        endif()
    endforeach()
endif()
