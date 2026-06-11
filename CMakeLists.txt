cmake_minimum_required(VERSION 3.16)
project(vlacpp VERSION 0.1.0 LANGUAGES C CXX)

option(VLACPP_BUILD_EXAMPLES "Build vlacpp examples" ON)
option(VLACPP_BUILD_TESTS "Build vlacpp tests" ON)
option(VLACPP_BUILD_SMOLVLA "Build SmolVLA migration targets" ON)

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

if(VLACPP_BUILD_SMOLVLA)
    set(SMOLVLA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/models/smolvla)
    set(SMOLVLA_ROBOT_SERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/robot_server)
    add_library(smolvla OBJECT
        ${SMOLVLA_DIR}/smolvla_engine.cpp
        ${SMOLVLA_DIR}/smolvla_engine.h
        ${SMOLVLA_DIR}/state_proj.cpp
        ${SMOLVLA_DIR}/state_proj.h
        ${SMOLVLA_DIR}/vision.cpp
        ${SMOLVLA_DIR}/vision.h
        ${SMOLVLA_DIR}/action_expert.cpp
        ${SMOLVLA_DIR}/action_expert.h
    )
    target_include_directories(smolvla
        PUBLIC
            ${SMOLVLA_DIR}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/stb
    )
    target_link_libraries(smolvla PRIVATE ggml llama)
    target_compile_features(smolvla PRIVATE cxx_std_17)
    if(NOT MSVC)
        target_compile_options(smolvla PRIVATE -Wno-cast-qual)
    endif()

    add_library(smolvla_static STATIC $<TARGET_OBJECTS:smolvla>)
    target_link_libraries(smolvla_static PUBLIC ggml llama)
    target_include_directories(smolvla_static PUBLIC ${SMOLVLA_DIR})

    add_library(robotcpp STATIC
        src/models/model.h
        src/models/model_factory.cpp
        ${SMOLVLA_DIR}/smolvla_model.cpp
        ${SMOLVLA_DIR}/smolvla_model.h
    )
    target_include_directories(robotcpp
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${SMOLVLA_DIR}
    )
    target_link_libraries(robotcpp PUBLIC smolvla_static)
    target_compile_features(robotcpp PUBLIC cxx_std_17)

    add_library(smolvla_robot_common STATIC
        ${SMOLVLA_ROBOT_SERVER_DIR}/protocol.cpp
        ${SMOLVLA_ROBOT_SERVER_DIR}/protocol.h
        ${SMOLVLA_ROBOT_SERVER_DIR}/socket.cpp
        ${SMOLVLA_ROBOT_SERVER_DIR}/socket.h
    )
    target_include_directories(smolvla_robot_common PUBLIC ${SMOLVLA_ROBOT_SERVER_DIR})
    target_compile_features(smolvla_robot_common PUBLIC cxx_std_17)
    if(WIN32)
        target_link_libraries(smolvla_robot_common PUBLIC ws2_32)
    endif()

    add_library(smolvla_robot_server STATIC
        ${SMOLVLA_ROBOT_SERVER_DIR}/session.cpp
        ${SMOLVLA_ROBOT_SERVER_DIR}/session.h
        ${SMOLVLA_ROBOT_SERVER_DIR}/model_adapter.h
    )
    target_include_directories(smolvla_robot_server PUBLIC ${SMOLVLA_ROBOT_SERVER_DIR})
    target_link_libraries(smolvla_robot_server PUBLIC smolvla_robot_common)
    target_compile_features(smolvla_robot_server PUBLIC cxx_std_17)

    add_library(model_client_cpp STATIC
        ${SMOLVLA_ROBOT_SERVER_DIR}/client/cpp/model_client.cpp
        ${SMOLVLA_ROBOT_SERVER_DIR}/client/cpp/model_client.h
    )
    target_include_directories(model_client_cpp PUBLIC ${SMOLVLA_ROBOT_SERVER_DIR})
    target_link_libraries(model_client_cpp PUBLIC smolvla_robot_common)
    target_compile_features(model_client_cpp PUBLIC cxx_std_17)

    add_executable(model-cli src/model-cli.cpp)
    set_target_properties(model-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(model-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/stb
    )
    target_link_libraries(model-cli PRIVATE robotcpp ggml llama)
    target_compile_features(model-cli PRIVATE cxx_std_17)

    add_executable(model-server
        ${SMOLVLA_ROBOT_SERVER_DIR}/model-server.cpp
        ${SMOLVLA_ROBOT_SERVER_DIR}/model_adapter.cpp
    )
    set_target_properties(model-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(model-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${SMOLVLA_DIR})
    target_link_libraries(model-server PRIVATE robotcpp smolvla_robot_server ggml llama)
    target_compile_features(model-server PRIVATE cxx_std_17)

    add_executable(model-cpp-client-example ${SMOLVLA_ROBOT_SERVER_DIR}/examples/cpp/minimal_predict.cpp)
    set_target_properties(model-cpp-client-example PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_link_libraries(model-cpp-client-example PRIVATE model_client_cpp)
    target_compile_features(model-cpp-client-example PRIVATE cxx_std_17)

    add_executable(smolvla-raw-predict ${SMOLVLA_ROBOT_SERVER_DIR}/test/smolvla_raw_predict.cpp)
    set_target_properties(smolvla-raw-predict PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_link_libraries(smolvla-raw-predict PRIVATE smolvla_static ggml llama)
    target_compile_features(smolvla-raw-predict PRIVATE cxx_std_17)
endif()

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
