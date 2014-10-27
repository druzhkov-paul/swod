set(target "gtest")

FILE(GLOB hdrs "gtest/*.h")
FILE(GLOB srcs "gtest/*.cc")

ADD_LIBRARY(${target} STATIC ${srcs})

TARGET_LINK_LIBRARIES(${target} ${Pthread})
