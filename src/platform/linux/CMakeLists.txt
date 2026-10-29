# Retain the existing POSIX compatibility implementation without duplicating it.
# Reuses POSIX bodies for Linux host development/testing; production CUDA, 4080, fit, and performance remain unqualified.
target_sources(saga_platform PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../posix/file_mapping.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../posix/process.cpp")
