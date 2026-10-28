# Retain the existing POSIX compatibility implementation without duplicating it.
# Linux build/packaging/performance qualification is a later workstream.
target_sources(saga_platform PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../posix/file_mapping.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../posix/process.cpp")
