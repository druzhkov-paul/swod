set(target ${SWOD_LIBRARY})

file(GLOB hdrs "*.h*" "${SWOD_INCLUDE}/swod/*.h*")
file(GLOB srcs "*.cpp")

#add_library(${target} SHARED ${srcs} ${hdrs})
add_library(${target} STATIC ${srcs} ${hdrs})
target_link_libraries(${target} ${LIBRARY_DEPS})
