# The ASR comparison harness (asr_host) against a whisper.cpp checkout, pinned in .github/workflows/asr-compare.yml to
# the version the kngfsh-translate app ships (v1.9.5). CPU only, as on the phone.
cmake_minimum_required(VERSION 3.14)
project(kngfsh_asr_host CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED WHISPER_DIR)
    message(FATAL_ERROR "set -DWHISPER_DIR=<whisper.cpp checkout>")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# Examples on for whisper-quantize (build only the targets you need: asr_host whisper-quantize).
set(WHISPER_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${WHISPER_DIR}" whisper_build)

add_executable(asr_host asr_host.cpp)
target_link_libraries(asr_host PRIVATE whisper)
