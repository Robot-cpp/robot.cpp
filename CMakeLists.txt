cmake_minimum_required(VERSION 3.16)
project(robotcpp VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

option(ROBOT_CPP_BUILD_ROBOT_SERVER "Build model-server target" ON)
option(ROBOT_CPP_BUILD_MODEL_CLI "Build model-cli target" OFF)
option(ROBOT_CPP_BUILD_ROBOT_CLIENT "Build C++ robot client targets" OFF)
option(ROBOT_CPP_BUILD_STARVLA "Build the StarVLA runtime (requires llama.cpp overlay)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "third_party/llama.cpp is required; run `git submodule update --init --recursive`")
endif()
if(ROBOT_CPP_BUILD_STARVLA)
    file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd/models/qwen3vl.cpp"
        ROBOT_CPP_QWEN3VL_MTMD_SOURCE)
    file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include/llama.h"
        ROBOT_CPP_LLAMA_PUBLIC_HEADER)
    string(FIND "${ROBOT_CPP_QWEN3VL_MTMD_SOURCE}"
        "FFN_GELU_ERF" ROBOT_CPP_QWEN3VL_PARITY_PATCH_INDEX)
    string(FIND "${ROBOT_CPP_LLAMA_PUBLIC_HEADER}"
        "llama_set_backend_native_graphs_enabled" ROBOT_CPP_LLAMA_GRAPH_PATCH_INDEX)
    if(ROBOT_CPP_QWEN3VL_PARITY_PATCH_INDEX EQUAL -1 OR
       ROBOT_CPP_LLAMA_GRAPH_PATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
            "StarVLA requires the pinned llama.cpp overlay. Run "
            "`./tools/llama_cpp/apply_starvla_patches.sh` from the repository root, "
            "then configure again.")
    endif()
    unset(ROBOT_CPP_QWEN3VL_MTMD_SOURCE)
    unset(ROBOT_CPP_LLAMA_PUBLIC_HEADER)
endif()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
if(NOT TARGET ggml OR NOT TARGET llama)
    message(FATAL_ERROR "llama.cpp must provide ggml and llama targets")
endif()
if(ROBOT_CPP_BUILD_STARVLA)
    # mtmd normally inherits this variable when llama.cpp builds all tools.
    if(NOT LLAMA_INSTALL_VERSION)
        set(LLAMA_INSTALL_VERSION ${PROJECT_VERSION})
    endif()
    add_subdirectory(third_party/llama.cpp/tools/mtmd EXCLUDE_FROM_ALL)
    if(NOT TARGET mtmd)
        message(FATAL_ERROR "llama.cpp must provide the mtmd target for Qwen-VL")
    endif()
endif()
set(ROBOT_CPP_LLAMA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/stb
)

set(SMOLVLA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/models/smolvla)
set(STARVLA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/models/starvla)
set(ROBOT_SERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/robot_server)
set(ROBOT_CLIENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/robot_client)

add_library(robotcpp_model_common STATIC
    src/models/ggml_backend.cpp
    src/models/ggml_backend.h
    src/models/gguf_loader.cpp
    src/models/gguf_loader.h
    src/models/model_type.cpp
)
target_include_directories(robotcpp_model_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ROBOT_CPP_LLAMA_INCLUDE_DIRS}
)
target_link_libraries(robotcpp_model_common PUBLIC ggml)
target_compile_features(robotcpp_model_common PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(robotcpp_model_common PRIVATE -Wno-cast-qual)
endif()

add_library(smolvla_runtime STATIC
    ${SMOLVLA_DIR}/smolvla_engine.cpp
    ${SMOLVLA_DIR}/smolvla_engine.h
    ${SMOLVLA_DIR}/state_proj.cpp
    ${SMOLVLA_DIR}/state_proj.h
    ${SMOLVLA_DIR}/vision.cpp
    ${SMOLVLA_DIR}/vision.h
    ${SMOLVLA_DIR}/action_expert.cpp
    ${SMOLVLA_DIR}/action_expert.h
)
target_include_directories(smolvla_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SMOLVLA_DIR}
        ${ROBOT_CPP_LLAMA_INCLUDE_DIRS}
)
target_link_libraries(smolvla_runtime PUBLIC robotcpp_model_common llama)
target_compile_features(smolvla_runtime PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(smolvla_runtime PRIVATE -Wno-cast-qual)
endif()

add_library(pi0_engine STATIC
    src/models/pi0/types.h
    src/models/pi0/action.cpp
    src/models/pi0/action.h
    src/models/pi0/component_runtime.cpp
    src/models/pi0/component_runtime.h
    src/models/pi0/config.h
    src/models/pi0/load.cpp
    src/models/pi0/load.h
    src/models/pi0/pi0_context.h
    src/models/pi0/pi0_engine.cpp
    src/models/pi0/pi0_engine.h
    src/models/pi0/preprocess.cpp
    src/models/pi0/preprocess.h
    src/models/pi0/weights.cpp
    src/models/pi0/weights.h
    src/models/pi0/vlm.cpp
    src/models/pi0/vlm.h
)
target_include_directories(pi0_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ROBOT_CPP_LLAMA_INCLUDE_DIRS}
)
target_link_libraries(pi0_engine PUBLIC robotcpp_model_common llama)
target_compile_features(pi0_engine PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(pi0_engine PRIVATE -Wno-cast-qual)
endif()

if(ROBOT_CPP_BUILD_STARVLA)
add_library(starvla_runtime STATIC
    ${STARVLA_DIR}/fast_codec.cpp
    ${STARVLA_DIR}/fast_codec.h
    ${STARVLA_DIR}/fast_policy.cpp
    ${STARVLA_DIR}/fast_policy.h
    ${STARVLA_DIR}/groot_policy.cpp
    ${STARVLA_DIR}/groot_policy.h
    ${STARVLA_DIR}/groot_prompt.cpp
    ${STARVLA_DIR}/groot_prompt.h
    ${STARVLA_DIR}/normalization.cpp
    ${STARVLA_DIR}/normalization.h
    ${STARVLA_DIR}/oft_image_preprocess.cpp
    ${STARVLA_DIR}/oft_image_preprocess.h
    ${STARVLA_DIR}/oft_prompt.cpp
    ${STARVLA_DIR}/oft_prompt.h
    ${STARVLA_DIR}/oft_policy.cpp
    ${STARVLA_DIR}/oft_policy.h
    ${STARVLA_DIR}/pi_policy.cpp
    ${STARVLA_DIR}/pi_policy.h
    ${STARVLA_DIR}/pi_v3_policy.cpp
    ${STARVLA_DIR}/pi_v3_policy.h
    ${STARVLA_DIR}/policy_gguf.h
    ${STARVLA_DIR}/qwen3vl_bridge.cpp
    ${STARVLA_DIR}/qwen3vl_bridge.h
    ${STARVLA_DIR}/starvla_engine.cpp
    ${STARVLA_DIR}/starvla_engine.h
    third_party/llama.cpp/examples/gguf-hash/deps/sha256/sha256.c
)
target_include_directories(starvla_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ROBOT_CPP_LLAMA_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/examples/gguf-hash/deps
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/examples/gguf-hash/deps/sha256
)
target_link_libraries(starvla_runtime PUBLIC robotcpp_model_common llama mtmd)
target_compile_features(starvla_runtime PUBLIC cxx_std_17)
if(GGML_CUDA)
    enable_language(CUDA)
    target_sources(starvla_runtime PRIVATE ${STARVLA_DIR}/qwen_bf16_round_cuda.cu)
    target_compile_definitions(starvla_runtime PRIVATE ROBOTCPP_STARVLA_CUDA=1)
    set_property(TARGET starvla_runtime PROPERTY CUDA_STANDARD 17)
endif()
if(NOT MSVC)
    target_compile_options(starvla_runtime PRIVATE -Wno-cast-qual)
endif()
endif()

add_library(robotcpp STATIC
    src/models/argument_parse.h
    src/models/model.h
    src/models/model_factory.cpp
    src/models/pi0/pi0_model.cpp
    src/models/pi0/pi0_model.h
    ${SMOLVLA_DIR}/smolvla_model.cpp
    ${SMOLVLA_DIR}/smolvla_model.h
)
target_include_directories(robotcpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SMOLVLA_DIR}
)
target_link_libraries(robotcpp PUBLIC smolvla_runtime pi0_engine)
target_compile_features(robotcpp PUBLIC cxx_std_17)
if(ROBOT_CPP_BUILD_STARVLA)
    target_sources(robotcpp PRIVATE
        ${STARVLA_DIR}/starvla_model.cpp
        ${STARVLA_DIR}/starvla_model.h)
    target_link_libraries(robotcpp PUBLIC starvla_runtime)
    target_compile_definitions(robotcpp PUBLIC ROBOT_CPP_BUILD_STARVLA=1)
endif()

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)

    add_executable(robotcpp-model-type-test tests/model_type_test.cpp)
    target_link_libraries(robotcpp-model-type-test PRIVATE robotcpp)
    target_compile_features(robotcpp-model-type-test PRIVATE cxx_std_17)
    add_test(NAME robotcpp-model-type-test COMMAND robotcpp-model-type-test)

    foreach(model_name IN ITEMS smolvla pi0)
        add_executable(robotcpp-${model_name}-model-test
            tests/${model_name}/model_test.cpp)
        target_link_libraries(robotcpp-${model_name}-model-test PRIVATE robotcpp)
        add_test(NAME robotcpp-${model_name}-model-test
            COMMAND robotcpp-${model_name}-model-test)
    endforeach()

    if(ROBOT_CPP_BUILD_STARVLA)
        foreach(test_name IN ITEMS oft_prompt groot_prompt oft_image_preprocess qwen_vl_profile fast_runtime)
            string(REPLACE "_" "-" target_name "${test_name}")
            add_executable(robotcpp-starvla-${target_name}-test
                tests/starvla/${test_name}_test.cpp)
            target_link_libraries(robotcpp-starvla-${target_name}-test
                PRIVATE starvla_runtime)
            target_compile_features(robotcpp-starvla-${target_name}-test
                PRIVATE cxx_std_17)
            add_test(NAME robotcpp-starvla-${target_name}-test
                COMMAND robotcpp-starvla-${target_name}-test)
        endforeach()

        add_executable(robotcpp-starvla-fast-codec-test
            tests/starvla/fast_codec_test.cpp)
        target_link_libraries(robotcpp-starvla-fast-codec-test PRIVATE starvla_runtime)
        target_compile_features(robotcpp-starvla-fast-codec-test PRIVATE cxx_std_17)
        add_test(NAME robotcpp-starvla-fast-codec-test
            COMMAND robotcpp-starvla-fast-codec-test)
    endif()
endif()

if(ROBOT_CPP_BUILD_ROBOT_SERVER OR ROBOT_CPP_BUILD_ROBOT_CLIENT)
    add_library(robot_server_common STATIC
        ${ROBOT_SERVER_DIR}/protocol.cpp
        ${ROBOT_SERVER_DIR}/protocol.h
        ${ROBOT_SERVER_DIR}/socket.cpp
        ${ROBOT_SERVER_DIR}/socket.h
    )
    target_include_directories(robot_server_common PUBLIC ${ROBOT_SERVER_DIR})
    target_compile_features(robot_server_common PUBLIC cxx_std_17)
    if(WIN32)
        target_link_libraries(robot_server_common PUBLIC ws2_32)
    endif()
    if(BUILD_TESTING)
        add_executable(robotcpp-protocol-test tests/model_protocol_test.cpp)
        target_link_libraries(robotcpp-protocol-test PRIVATE robot_server_common)
        add_test(NAME robotcpp-protocol-test COMMAND robotcpp-protocol-test)
    endif()
endif()

if(ROBOT_CPP_BUILD_ROBOT_CLIENT)
    add_library(model_client_cpp STATIC
        ${ROBOT_CLIENT_DIR}/cpp/model_client.cpp
        ${ROBOT_CLIENT_DIR}/cpp/model_client.h
    )
    target_include_directories(model_client_cpp PUBLIC
        ${ROBOT_CLIENT_DIR}
        ${ROBOT_SERVER_DIR}
    )
    target_link_libraries(model_client_cpp PUBLIC robot_server_common)
    target_compile_features(model_client_cpp PUBLIC cxx_std_17)

    add_executable(model-cpp-client-example ${ROBOT_CLIENT_DIR}/examples/cpp/minimal_example.cpp)
    set_target_properties(model-cpp-client-example PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_link_libraries(model-cpp-client-example PRIVATE model_client_cpp)
    target_compile_features(model-cpp-client-example PRIVATE cxx_std_17)

endif()

if(ROBOT_CPP_BUILD_ROBOT_SERVER)
    add_library(robot_server_core STATIC
        ${ROBOT_SERVER_DIR}/session.cpp
        ${ROBOT_SERVER_DIR}/session.h
        ${ROBOT_SERVER_DIR}/model_adapter.cpp
        ${ROBOT_SERVER_DIR}/model_adapter.h
    )
    target_include_directories(robot_server_core
        PUBLIC
            ${ROBOT_SERVER_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(robot_server_core PUBLIC robot_server_common robotcpp)
    target_compile_features(robot_server_core PUBLIC cxx_std_17)

    add_executable(model-server ${ROBOT_SERVER_DIR}/model-server.cpp)
    set_target_properties(model-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(model-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SMOLVLA_DIR}
        ${ROBOT_SERVER_DIR}
    )
    target_link_libraries(model-server PRIVATE robot_server_core robotcpp)
    target_compile_features(model-server PRIVATE cxx_std_17)
    if(BUILD_TESTING AND Python3_Interpreter_FOUND)
        add_test(NAME robotcpp-model-server-args-test
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/model_frontend_args_test.py
                --kind server --executable $<TARGET_FILE:model-server>)
    endif()

    add_executable(smolvla-raw-predict ${ROBOT_SERVER_DIR}/test/smolvla_raw_predict.cpp)
    set_target_properties(smolvla-raw-predict PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(smolvla-raw-predict PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${SMOLVLA_DIR})
    target_link_libraries(smolvla-raw-predict PRIVATE smolvla_runtime)
    target_compile_features(smolvla-raw-predict PRIVATE cxx_std_17)
endif()

if(ROBOT_CPP_BUILD_MODEL_CLI)
    add_executable(model-cli src/model-cli.cpp)
    set_target_properties(model-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(model-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ROBOT_CPP_LLAMA_INCLUDE_DIRS}
    )
    target_link_libraries(model-cli PRIVATE robotcpp)
    target_compile_features(model-cli PRIVATE cxx_std_17)
    if(BUILD_TESTING AND Python3_Interpreter_FOUND)
        add_test(NAME robotcpp-model-cli-args-test
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/model_frontend_args_test.py
                --kind cli --executable $<TARGET_FILE:model-cli>)
    endif()
endif()
