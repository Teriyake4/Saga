cmake_minimum_required(VERSION 3.28)
project(Saga VERSION 0.1.0 LANGUAGES CXX)

# Saga has no C++ modules; avoid requiring clang-scan-deps during configure try-compiles.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

option(SAGA_BUILD_TESTS "Build host tests (no model weights required)" ON)
option(SAGA_ENABLE_CUDA "Build the SM89 CUDA backend and portability probe" OFF)
option(SAGA_BUILD_CUDA_TESTS "Build physical CUDA validation executables (requires CUDA)" OFF)
option(SAGA_ENABLE_IMAGE_DECODER "Build the optional bounded PNG/JPEG image decoder" OFF)
option(SAGA_WARNINGS_AS_ERRORS "Treat warnings in Saga sources as errors" ON)
option(SAGA_SANITIZERS "Enable host address and undefined behavior sanitizers" OFF)
option(SAGA_THREAD_SANITIZER "Enable host thread sanitizer (separate from ASan/UBSan)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

add_library(saga_build_options INTERFACE)
if(MSVC)
  target_compile_options(saga_build_options INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/utf-8;/Zc:__cplusplus>")
  if(SAGA_WARNINGS_AS_ERRORS)
    target_compile_options(saga_build_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
  target_compile_definitions(saga_build_options INTERFACE WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00)
else()
  target_compile_options(saga_build_options INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
  if(SAGA_WARNINGS_AS_ERRORS)
    target_compile_options(saga_build_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()
if(SAGA_BUILD_CUDA_TESTS AND NOT SAGA_ENABLE_CUDA)
  message(FATAL_ERROR "SAGA_BUILD_CUDA_TESTS requires SAGA_ENABLE_CUDA=ON")
endif()
if(SAGA_ENABLE_CUDA AND (SAGA_SANITIZERS OR SAGA_THREAD_SANITIZER))
  message(FATAL_ERROR "Host sanitizer configurations are separate from CUDA validation")
endif()
if(SAGA_SANITIZERS AND SAGA_THREAD_SANITIZER)
  message(FATAL_ERROR "ThreadSanitizer must run separately from AddressSanitizer/UBSan")
endif()
if(SAGA_SANITIZERS OR SAGA_THREAD_SANITIZER)
  if(MSVC OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "Host sanitizers currently require a Clang/GCC toolchain")
  endif()
  if(SAGA_THREAD_SANITIZER)
    set(saga_sanitizer_flags -fsanitize=thread)
  else()
    set(saga_sanitizer_flags -fsanitize=address,undefined)
  endif()
  target_compile_options(saga_build_options INTERFACE ${saga_sanitizer_flags} -fno-omit-frame-pointer)
  target_link_options(saga_build_options INTERFACE ${saga_sanitizer_flags})
endif()

add_library(saga_dependencies INTERFACE)
target_include_directories(saga_dependencies SYSTEM INTERFACE
  "${PROJECT_SOURCE_DIR}/third_party/cpp-httplib"
  "${PROJECT_SOURCE_DIR}/third_party/json"
  "${PROJECT_SOURCE_DIR}/third_party/tomlplusplus")
target_compile_definitions(saga_dependencies INTERFACE
  CPPHTTPLIB_HEADER_MAX_COUNT=64 CPPHTTPLIB_HEADER_MAX_LENGTH=8192
  CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=4096)

add_subdirectory(src/platform)

if(SAGA_ENABLE_IMAGE_DECODER)
  add_library(saga_image_decoder src/media/image_decoder.cpp)
  target_include_directories(saga_image_decoder PUBLIC include)
  target_include_directories(saga_image_decoder SYSTEM PRIVATE third_party/stb-image)
  target_link_libraries(saga_image_decoder PRIVATE saga_build_options)
endif()

add_library(saga_foundation
  src/backend/contracts.cpp src/backend/device_profile.cpp src/memory/ledger.cpp src/memory/planner.cpp src/kv/catalog.cpp)
target_include_directories(saga_foundation PUBLIC include)
target_link_libraries(saga_foundation PUBLIC saga_build_options saga_platform Threads::Threads)

add_library(saga_runtime
  src/runtime/completion.cpp
  src/runtime/execution_worker.cpp
  src/runtime/sequence_slots.cpp
  src/runtime/admission.cpp
  src/runtime/output_buffer.cpp
  src/runtime/model_contracts.cpp
  src/runtime/startup.cpp)
target_include_directories(saga_runtime PUBLIC include)
target_link_libraries(saga_runtime PUBLIC saga_foundation)

add_library(saga_host src/core/config.cpp src/core/model_config.cpp src/core/log.cpp src/api/server.cpp)
target_include_directories(saga_host PUBLIC include)
target_link_libraries(saga_host PUBLIC saga_foundation PRIVATE saga_dependencies)
if(WIN32)
  target_link_libraries(saga_host PRIVATE ws2_32)
endif()

add_executable(saga-server apps/server/main.cpp)
target_link_libraries(saga-server PRIVATE saga_host)

if(SAGA_ENABLE_CUDA)
  if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux|Windows")
    message(FATAL_ERROR "The production CUDA target requires Linux or native Windows; host-only builds remain supported")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR "The initial SM89 reference-toolchain floor is CUDA 12.8; qualification is still required")
  endif()
  add_executable(saga_cuda_portability_smoke tests/cuda/portability_smoke.cu)
  set_target_properties(saga_cuda_portability_smoke PROPERTIES
    CUDA_ARCHITECTURES 89 CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(saga_cuda_portability_smoke PRIVATE CUDA::cudart)
  if(WIN32)
    target_compile_definitions(saga_cuda_portability_smoke PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  endif()

  add_library(saga_cuda STATIC
    src/backends/cuda/projection.cu src/backends/cuda/resources.cpp)
  target_include_directories(saga_cuda PUBLIC include)
  target_link_libraries(saga_cuda PUBLIC saga_foundation PRIVATE CUDA::cudart CUDA::cuda_driver)
  set_target_properties(saga_cuda PROPERTIES
    CUDA_ARCHITECTURES 89 CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON CUDA_RUNTIME_LIBRARY Shared)

  if(SAGA_BUILD_CUDA_TESTS)
    add_executable(saga_cuda_projection_test tests/cuda/projection_test.cu)
    target_link_libraries(saga_cuda_projection_test PRIVATE saga_cuda saga_dependencies CUDA::cudart)
    set_target_properties(saga_cuda_projection_test PROPERTIES
      CUDA_ARCHITECTURES 89 CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Shared)
  endif()
endif()

if(SAGA_BUILD_TESTS)
  enable_testing()
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
  add_executable(saga_cuda_resource_owners_test tests/cuda_resource_owners_test.cpp)
  target_include_directories(saga_cuda_resource_owners_test PRIVATE tests/cuda/fake_runtime src/backends/cuda)
  target_link_libraries(saga_cuda_resource_owners_test PRIVATE saga_build_options)
  add_test(NAME cuda_resource_owners COMMAND saga_cuda_resource_owners_test)
  set_tests_properties(cuda_resource_owners PROPERTIES TIMEOUT 45)
  foreach(name contracts memory resource_planner platform platform_process config model_config completion execution_worker sequence_slots admission output_buffer runtime_integration model_contracts)
    add_executable(saga_${name}_test tests/${name}_test.cpp)
    if(name STREQUAL "config" OR name STREQUAL "model_config")
      target_link_libraries(saga_${name}_test PRIVATE saga_host)
    elseif(name STREQUAL "completion" OR name STREQUAL "execution_worker" OR name STREQUAL "sequence_slots"
            OR name STREQUAL "admission" OR name STREQUAL "output_buffer" OR name STREQUAL "runtime_integration"
                        OR name STREQUAL "model_contracts")
      target_link_libraries(saga_${name}_test PRIVATE saga_runtime)
    else()
      target_link_libraries(saga_${name}_test PRIVATE saga_foundation)
    endif()
    add_test(NAME ${name} COMMAND saga_${name}_test)
    set_tests_properties(${name} PROPERTIES TIMEOUT 45)
  endforeach()
  add_executable(saga_startup_test tests/startup_test.cpp)
  target_link_libraries(saga_startup_test PRIVATE saga_runtime)
  add_test(NAME startup COMMAND saga_startup_test)
  set_tests_properties(startup PROPERTIES TIMEOUT 45)
  add_executable(saga_step_test tests/step_test.cpp)
  target_link_libraries(saga_step_test PRIVATE saga_runtime)
  add_test(NAME step COMMAND saga_step_test)
  set_tests_properties(step PROPERTIES TIMEOUT 45)
  add_executable(saga_device_profile_test tests/device_profile_test.cpp)
  target_link_libraries(saga_device_profile_test PRIVATE saga_foundation)
  add_test(NAME device_profile COMMAND saga_device_profile_test)
  set_tests_properties(device_profile PROPERTIES TIMEOUT 45)
  # Isolated vendor parser/accept-loop probes; synthetic accept wrappers must not
  # share a translation unit or link target with the production HTTP adapter.
  add_executable(saga_http_transport_test tests/http_transport_test.cpp)
  target_link_libraries(saga_http_transport_test PRIVATE saga_dependencies saga_build_options Threads::Threads)
  if(WIN32)
    target_link_libraries(saga_http_transport_test PRIVATE ws2_32)
  endif()
  add_test(NAME http_transport COMMAND saga_http_transport_test)
  if(SAGA_ENABLE_IMAGE_DECODER)
    add_executable(saga_image_decoder_test tests/image_decoder_test.cpp)
    target_link_libraries(saga_image_decoder_test PRIVATE saga_image_decoder saga_build_options Threads::Threads)
    add_test(NAME image_decoder COMMAND saga_image_decoder_test)
    set_tests_properties(image_decoder PROPERTIES TIMEOUT 45)
  endif()
  set_tests_properties(http_transport PROPERTIES TIMEOUT 45)
  add_test(NAME api_shell COMMAND "${Python3_EXECUTABLE}" -B "${PROJECT_SOURCE_DIR}/tests/api/shell_test.py"
    --server "$<TARGET_FILE:saga-server>" --config "${PROJECT_SOURCE_DIR}/config/server.example.toml")
  set_tests_properties(api_shell PROPERTIES TIMEOUT 90)
  if(WIN32)
    add_executable(saga_windows_console_test tests/platform/windows_console_test.cpp)
    target_link_libraries(saga_windows_console_test PRIVATE saga_platform)
    add_test(NAME windows_console_signals COMMAND "${Python3_EXECUTABLE}" -B
      "${PROJECT_SOURCE_DIR}/tests/platform/windows_console_test.py" "$<TARGET_FILE:saga_windows_console_test>")
    set_tests_properties(windows_console_signals PROPERTIES TIMEOUT 60)
  endif()
  add_test(NAME m1_evidence_tests COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover
    -s "${PROJECT_SOURCE_DIR}/tests/tooling" -v)
  add_test(NAME dependency_hashes COMMAND "${Python3_EXECUTABLE}" -B "${PROJECT_SOURCE_DIR}/tools/verify_dependencies.py")
  add_test(NAME source_manifest_structure COMMAND "${Python3_EXECUTABLE}" -B
    "${PROJECT_SOURCE_DIR}/tools/verify_sources.py" --structure-only)
  add_test(NAME source_verifier_tests COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover
    -s "${PROJECT_SOURCE_DIR}/bench/tests" -v)
  set_tests_properties(m1_evidence_tests PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 60)
  set_tests_properties(dependency_hashes source_manifest_structure source_verifier_tests PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 60)
endif()

include(GNUInstallDirs)
install(TARGETS saga-server RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES config/server.example.toml config/model.example.ini DESTINATION "${CMAKE_INSTALL_DATADIR}/saga")
install(FILES third_party/cpp-httplib/LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/saga/licenses" RENAME cpp-httplib.LICENSE)
install(FILES third_party/json/LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/saga/licenses" RENAME json.LICENSE)
install(FILES third_party/tomlplusplus/LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/saga/licenses" RENAME tomlplusplus.LICENSE)
if(SAGA_ENABLE_IMAGE_DECODER)
  install(TARGETS saga_image_decoder ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(FILES include/saga/media/image_decoder.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/saga/media")
  install(FILES third_party/stb-image/LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/saga/licenses" RENAME stb_image.LICENSE)
endif()
