cmake_minimum_required(VERSION 3.16)
project(vlacpp VERSION 0.1.0 LANGUAGES C CXX)

option(VLACPP_BUILD_EXAMPLES "Build vlacpp examples" ON)
option(VLACPP_BUILD_TESTS "Build vlacpp tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vlacpp SHARED
    src/core/context.cpp
    src/core/error.cpp
    src/core/gguf.cpp
    src/core/json.cpp
    src/core/preprocess.cpp
    src/models/model.cpp
    src/models/ggml_runtime.cpp
    src/models/pi0/pi0_action_expert.cpp
    src/models/pi0/pi0_action_decoder.cpp
    src/models/ggml_ops.cpp
    src/models/pi0/pi0_language_prefix.cpp
    src/models/pi0/pi0_sampler.cpp
    src/models/pi0/pi0_tokenizer.cpp
    src/models/pi0/pi0_vision_mtmd.cpp
    src/models/pi0/pi0_vlm.cpp
    src/models/pi0/pi0.cpp
    src/sampling/flow.cpp
)

target_include_directories(vlacpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
)

target_compile_features(vlacpp PUBLIC cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "third_party/llama.cpp is required; run `git submodule update --init --recursive`")
endif()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
if(NOT TARGET ggml OR NOT TARGET llama OR NOT TARGET mtmd)
    message(FATAL_ERROR "llama.cpp must provide ggml, llama, and mtmd targets")
endif()
target_link_libraries(vlacpp PRIVATE ggml llama mtmd)
target_compile_definitions(vlacpp PRIVATE VLACPP_HAVE_GGML=1 VLACPP_HAVE_MTMD=1)

if(VLACPP_BUILD_EXAMPLES)
    add_executable(vlacpp-pi0 examples/pi0-cli/main.cpp)
    target_link_libraries(vlacpp-pi0 PRIVATE vlacpp)
endif()

if(VLACPP_BUILD_TESTS)
    enable_testing()
    add_executable(vlacpp-tests tests/test_runtime.cpp)
    target_link_libraries(vlacpp-tests PRIVATE vlacpp)
    add_test(NAME vlacpp-runtime COMMAND vlacpp-tests)
    add_executable(vlacpp-gguf-tests tests/test_gguf_core.cpp)
    target_include_directories(vlacpp-gguf-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vlacpp-gguf-tests PRIVATE vlacpp)
    add_test(NAME vlacpp-gguf-core COMMAND vlacpp-gguf-tests)
    add_executable(vlacpp-mtmd-tests tests/test_mtmd_api.cpp)
    target_link_libraries(vlacpp-mtmd-tests PRIVATE vlacpp mtmd)
    add_test(NAME vlacpp-mtmd-api COMMAND vlacpp-mtmd-tests)
    add_executable(vlacpp-pi0-action-expert-tests tests/test_pi0_action_expert.cpp)
    target_include_directories(vlacpp-pi0-action-expert-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vlacpp-pi0-action-expert-tests PRIVATE vlacpp)
    add_test(NAME vlacpp-pi0-action-expert COMMAND vlacpp-pi0-action-expert-tests)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(VLACPP_ACTION_HEAD_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/tests/action-head-checkpoint.json")
        set(VLACPP_ACTION_HEAD_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head.gguf")
        set(VLACPP_ACTION_HEAD_SAFETENSORS "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-names.safetensors")
        set(VLACPP_ACTION_HEAD_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-map.json")
        set(VLACPP_ACTION_HEAD_MAPPED_GGUF "${CMAKE_CURRENT_BINARY_DIR}/action-head-openpi-mapped.gguf")
        add_test(
            NAME vlacpp-require-full-rejects-restricted
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_require_full.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
        )
        add_test(
            NAME vlacpp-python-api
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_api.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --library "$<TARGET_FILE:vlacpp>"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-gguf-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --output "${VLACPP_ACTION_HEAD_GGUF}"
        )
        set_tests_properties(vlacpp-require-full-rejects-restricted PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)
        set_tests_properties(vlacpp-python-api PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)
        add_test(
            NAME vlacpp-action-head-cli
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli_actions.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
                --horizon 3
                --action-dim 2
                --capability "restricted-pi0-state-action-head"
                --graph-action-width 4
        )
        add_test(
            NAME vlacpp-action-head-gguf-inspect
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_gguf_tensors.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_action_head_reference.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_ACTION_HEAD_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        set_tests_properties(vlacpp-action-head-cli PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)
        set_tests_properties(vlacpp-action-head-gguf-inspect PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)
        set_tests_properties(vlacpp-action-head-reference-compare PROPERTIES DEPENDS vlacpp-action-head-gguf-convert)

        add_test(
            NAME vlacpp-action-head-safetensors-create
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_action_head_safetensors.py"
                "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
        )
        set_tests_properties(vlacpp-action-head-safetensors-create PROPERTIES SKIP_RETURN_CODE 77)
        add_test(
            NAME vlacpp-action-head-safetensors-map
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/map-openpi-tensors.py"
                "${VLACPP_ACTION_HEAD_SAFETENSORS}"
                --family action-expert
                --require-complete
                --output "${VLACPP_ACTION_HEAD_MANIFEST}"
        )
        add_test(
            NAME vlacpp-action-head-mapped-safetensors-convert
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-openpi-to-gguf.py"
                --tensor-map-manifest "${VLACPP_ACTION_HEAD_MANIFEST}"
                --output "${VLACPP_ACTION_HEAD_MAPPED_GGUF}"
        )
        add_test(
            NAME vlacpp-action-head-mapped-safetensors-reference-compare
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_action_head_reference.py"
                --checkpoint "${VLACPP_ACTION_HEAD_CHECKPOINT}"
                --model "${VLACPP_ACTION_HEAD_MAPPED_GGUF}"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --state "1,-2"
                --prompt "pick up the fork"
                --steps 2
                --seed 1
        )
        set_tests_properties(vlacpp-action-head-safetensors-map PROPERTIES DEPENDS vlacpp-action-head-safetensors-create)
        set_tests_properties(vlacpp-action-head-mapped-safetensors-convert PROPERTIES DEPENDS vlacpp-action-head-safetensors-map)
        set_tests_properties(vlacpp-action-head-mapped-safetensors-reference-compare PROPERTIES DEPENDS vlacpp-action-head-mapped-safetensors-convert)

    endif()
endif()
