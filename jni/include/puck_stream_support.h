/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef OPENSEARCH_KNN_JNI_PUCK_STREAM_SUPPORT_H
#define OPENSEARCH_KNN_JNI_PUCK_STREAM_SUPPORT_H

#include "jni_util.h"
#include "native_engines_stream_support.h"
#include "parameter_utils.h"

#include <jni.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace knn_jni {
namespace puck_stream {

/**
 * A glue component to write Puck index data to OpenSearch streams.
 * Since Puck doesn't have a native streaming API, we provide write methods
 * that forward to the OpenSearch IndexOutput mediator.
 */
class PuckOpenSearchIOWriter {
 public:
  explicit PuckOpenSearchIOWriter(stream::NativeEngineIndexOutputMediator *_mediator)
      : mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {
  }

  /**
   * Write raw bytes to the stream.
   * @param ptr Pointer to data to write
   * @param size Size in bytes to write
   * @return Number of bytes written
   */
  size_t write(const void *ptr, size_t size) {
    if (size > 0) {
      mediator->writeBytes(reinterpret_cast<const uint8_t *>(ptr), size);
    }
    return size;
  }

  /**
   * Write a 32-bit integer to the stream.
   */
  void writeInt(int32_t value) {
    write(&value, sizeof(value));
  }

  /**
   * Write a 64-bit integer to the stream.
   */
  void writeLong(int64_t value) {
    write(&value, sizeof(value));
  }

  /**
   * Write a 32-bit unsigned integer to the stream.
   */
  void writeUInt(uint32_t value) {
    write(&value, sizeof(value));
  }

  /**
   * Write a string (length-prefixed) to the stream.
   */
  void writeString(const std::string& str) {
    uint32_t len = str.length();
    writeUInt(len);
    write(str.data(), len);
  }

  /**
   * Flush any buffered data to the underlying stream.
   */
  void flush() {
    mediator->flush();
  }

 private:
  stream::NativeEngineIndexOutputMediator *mediator;
};  // class PuckOpenSearchIOWriter


/**
 * A glue component to read Puck index data from OpenSearch streams.
 */
class PuckOpenSearchIOReader {
 public:
  explicit PuckOpenSearchIOReader(stream::NativeEngineIndexInputMediator *_mediator)
      : mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {
  }

  /**
   * Read raw bytes from the stream.
   * @param ptr Pointer to buffer to read into
   * @param size Number of bytes to read
   * @return Number of bytes read
   */
  size_t read(void *ptr, size_t size) {
    if (size > 0) {
      mediator->copyBytes(size, reinterpret_cast<uint8_t *>(ptr));
    }
    return size;
  }

  /**
   * Read a 32-bit integer from the stream.
   */
  int32_t readInt() {
    int32_t value;
    read(&value, sizeof(value));
    return value;
  }

  /**
   * Read a 64-bit integer from the stream.
   */
  int64_t readLong() {
    int64_t value;
    read(&value, sizeof(value));
    return value;
  }

  /**
   * Read a 32-bit unsigned integer from the stream.
   */
  uint32_t readUInt() {
    uint32_t value;
    read(&value, sizeof(value));
    return value;
  }

  /**
   * Read a string (length-prefixed) from the stream.
   */
  std::string readString() {
    uint32_t len = readUInt();
    std::string str(len, '\0');
    read(&str[0], len);
    return str;
  }

 private:
  stream::NativeEngineIndexInputMediator *mediator;
};  // class PuckOpenSearchIOReader


/**
 * Helper functions for Puck stream I/O.
 * Since Puck uses file-based I/O, these helpers facilitate the temporary file approach.
 */

/**
 * Write contents of a file to OpenSearch stream
 *
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param output OpenSearch IndexOutput stream
 * @param filePath Path to file to read from
 */
void writeFileToStream(
    knn_jni::JNIUtilInterface* jniUtil,
    JNIEnv* env,
    jobject output,
    const std::string& filePath
);

/**
 * Read data from OpenSearch stream and write to file
 *
 * @param jniUtil JNI utility interface
 * @param env JNI environment
 * @param input OpenSearch IndexInput stream
 * @param filePath Path to file to write to
 * @param fileSize Size of data to read/write
 */
void readStreamToFile(
    knn_jni::JNIUtilInterface* jniUtil,
    JNIEnv* env,
    jobject input,
    const std::string& filePath,
    size_t fileSize
);

}  // namespace puck_stream
}  // namespace knn_jni

#endif //OPENSEARCH_KNN_JNI_PUCK_STREAM_SUPPORT_H
