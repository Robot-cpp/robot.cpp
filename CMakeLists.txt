cmake_minimum_required(VERSION 3.16)
project(robotcpp VERSION 0.1.0 LANGUAGES C CXX)

option(ROBOT_CPP_BUILD_ROBOT_SERVER "Build model-server target" ON)
option(ROBOT_CPP_BUILD_MODEL_CLI "Build model-cli target" OFF)
option(ROBOT_CPP_BUILD_ROBOT_CLIENT "Build C++ robot client targets" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

set(ROBOTCPP_LLAMA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/stb
)

set(SMOLVLA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/models/smolvla)
set(ROBOT_SERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/robot_server)
set(ROBOT_CLIENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/robot_client)

add_library(smolvla_runtime STATIC
    src/models/ggml_backend.cpp
    src/models/ggml_backend.h
    src/models/gguf_loader.cpp
    src/models/gguf_loader.h
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
        ${ROBOTCPP_LLAMA_INCLUDE_DIRS}
)
target_link_libraries(smolvla_runtime PUBLIC ggml llama)
target_compile_features(smolvla_runtime PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(smolvla_runtime PRIVATE -Wno-cast-qual)
endif()

add_library(pi0_engine STATIC
    src/models/ggml_backend.cpp
    src/models/ggml_backend.h
    src/models/gguf_loader.cpp
    src/models/gguf_loader.h
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
        ${ROBOTCPP_LLAMA_INCLUDE_DIRS}
)
target_link_libraries(pi0_engine PUBLIC ggml llama)
target_compile_features(pi0_engine PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(pi0_engine PRIVATE -Wno-cast-qual)
endif()

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
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SMOLVLA_DIR}
)
target_link_libraries(robotcpp PUBLIC smolvla_runtime pi0_engine)
target_compile_features(robotcpp PUBLIC cxx_std_17)

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
        ${ROBOTCPP_LLAMA_INCLUDE_DIRS}
    )
    target_link_libraries(model-cli PRIVATE robotcpp)
    target_compile_features(model-cli PRIVATE cxx_std_17)
endif()
