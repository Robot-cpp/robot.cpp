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
    src/core/preprocess.cpp
    src/models/component_load.cpp
    src/models/model.cpp
    src/models/ggml_runtime.cpp
    src/models/pi0/action.cpp
    src/models/pi0/load.cpp
    src/models/pi0/model.cpp
    src/models/pi0/weights.cpp
    src/models/pi0/vlm.cpp
    src/sampling/flow.cpp
)

target_include_directories(vlacpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(vlacpp PUBLIC cxx_std_17)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "third_party/llama.cpp is required; run `git submodule update --init --recursive`")
endif()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
if(NOT TARGET ggml OR NOT TARGET llama)
    message(FATAL_ERROR "llama.cpp must provide ggml and llama targets")
endif()
target_link_libraries(vlacpp PRIVATE ggml llama)
target_compile_definitions(vlacpp PRIVATE VLACPP_HAVE_GGML=1)

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
        src/models/pi0/pi0_model.cpp
        src/models/pi0/pi0_model.h
        ${SMOLVLA_DIR}/smolvla_model.cpp
        ${SMOLVLA_DIR}/smolvla_model.h
    )
    target_include_directories(robotcpp
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${SMOLVLA_DIR}
    )
    target_link_libraries(robotcpp PUBLIC smolvla_static vlacpp)
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
    set_tests_properties(vlacpp-runtime PROPERTIES LABELS "smoke;regression;core")
    add_executable(vlacpp-gguf-tests tests/test_gguf_core.cpp)
    target_include_directories(vlacpp-gguf-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vlacpp-gguf-tests PRIVATE vlacpp)
    add_test(NAME vlacpp-gguf-core COMMAND vlacpp-gguf-tests)
    set_tests_properties(vlacpp-gguf-core PROPERTIES LABELS "smoke;regression;core;gguf")
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(VLACPP_FAKE_PI0_DIR "${CMAKE_CURRENT_BINARY_DIR}/fake-pi0-components")
        set(VLACPP_FAKE_PI0_VIT "${VLACPP_FAKE_PI0_DIR}/fake-pi0.vit.gguf")
        set(VLACPP_FAKE_PI0_MMPROJ "${VLACPP_FAKE_PI0_DIR}/fake-pi0.mmproj.gguf")
        set(VLACPP_FAKE_PI0_LLM "${VLACPP_FAKE_PI0_DIR}/fake-pi0.llm.gguf")
        set(VLACPP_FAKE_PI0_TOKENIZER "${VLACPP_FAKE_PI0_DIR}/fake-pi0.tokenizer.gguf")
        set(VLACPP_FAKE_PI0_STATE "${VLACPP_FAKE_PI0_DIR}/fake-pi0.state.gguf")
        set(VLACPP_FAKE_PI0_ACTION_DECODER "${VLACPP_FAKE_PI0_DIR}/fake-pi0.action_decoder.gguf")
        add_test(
            NAME vlacpp-fake-pi0-components
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_fake_pi0_components.py"
                "${VLACPP_FAKE_PI0_DIR}"
        )
        add_test(
            NAME vlacpp-python-api
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_api.py"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --library "$<TARGET_FILE:vlacpp>"
                --components "${VLACPP_FAKE_PI0_DIR}"
        )
        set_tests_properties(vlacpp-fake-pi0-components PROPERTIES
            LABELS "smoke;regression;conversion;pi0"
            FIXTURES_SETUP fake_pi0_components)
        set_tests_properties(vlacpp-python-api PROPERTIES
            LABELS "regression;python;pi0"
            FIXTURES_REQUIRED fake_pi0_components)
        add_test(
            NAME vlacpp-fake-pi0-cli
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli_actions.py"
                --binary "$<TARGET_FILE:vlacpp-pi0>"
                --vit "${VLACPP_FAKE_PI0_VIT}"
                --mmproj "${VLACPP_FAKE_PI0_MMPROJ}"
                --llm "${VLACPP_FAKE_PI0_LLM}"
                --tokenizer "${VLACPP_FAKE_PI0_TOKENIZER}"
                --state-gguf "${VLACPP_FAKE_PI0_STATE}"
                --action-decoder "${VLACPP_FAKE_PI0_ACTION_DECODER}"
                --state "1,-2"
                --steps 2
                --seed 1
                --horizon 2
                --action-dim 2
                --info-action-dim 2
        )
        set_tests_properties(vlacpp-fake-pi0-cli PROPERTIES
            LABELS "regression;pi0"
            FIXTURES_REQUIRED fake_pi0_components)
    endif()
endif()
