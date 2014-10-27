# get all cpp-files in current directory
file(GLOB apps_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(app_filename ${apps_list})
  # get file name without extension
  get_filename_component(app ${app_filename} NAME_WE)
  # generate project name
  set(app_name "app_${app}")
  # add and configure executable file to be produced 
  add_executable(${app_name} ${app_filename})
  target_link_libraries(${app_name} ${SWOD_LIBRARY})
  set_target_properties(${app_name} PROPERTIES
    OUTPUT_NAME "${app_name}"
    PROJECT_LABEL "${app_name}")
endforeach()
