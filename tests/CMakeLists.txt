set(target "${SWOD_TESTS}")

FILE(GLOB hdrs "*.h*")
FILE(GLOB srcs "*.cpp")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty")

ADD_EXECUTABLE(${target} ${srcs} ${hdrs})
TARGET_LINK_LIBRARIES(${target} gtest ${SWOD_LIBRARY})


if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    add_custom_command(
        TARGET ${SWOD_TESTS}
        POST_BUILD
        COMMAND cp -R "${CMAKE_SOURCE_DIR}/tests/testdata/" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}"
        COMMENT "Coping testdata"
    )
else()
    add_custom_command(
        TARGET ${SWOD_TESTS}
        POST_BUILD
        COMMAND ln -fns "${CMAKE_SOURCE_DIR}/tests/testdata/" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}"
        COMMENT "Adding a symbolic link to testdata"
    )
endif()
