add_library(saga_platform
  shared/file_mapping.cpp
  shared/process.cpp)
target_include_directories(saga_platform PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(saga_platform PUBLIC saga_build_options Threads::Threads)

if(WIN32)
  add_subdirectory(windows)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Compatibility selection only: Linux release qualification is deferred.
  add_subdirectory(linux)
elseif(APPLE)
  # Development host regression tests, not a macOS inference backend.
  add_subdirectory(posix)
else()
  message(FATAL_ERROR "No Saga native platform implementation for ${CMAKE_SYSTEM_NAME}")
endif()
