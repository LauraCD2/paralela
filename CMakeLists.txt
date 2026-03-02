cmake_minimum_required(VERSION 3.16)

project(paralela LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(paralela src/main.cpp)

# SFML is a C++ library, so ensure C++ linker behavior even with C sources.
set_target_properties(paralela PROPERTIES LINKER_LANGUAGE CXX)

if(MSVC)
  target_compile_options(paralela PRIVATE /openmp)
  target_link_libraries(
    paralela
    PRIVATE
      sfml-graphics.lib
      sfml-window.lib
      sfml-system.lib
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(paralela PRIVATE -fopenmp)
  target_link_options(paralela PRIVATE -fopenmp)
  target_link_libraries(
    paralela
    PRIVATE
      sfml-graphics
      sfml-window
      sfml-system
  )
else()
  message(WARNING "Unknown compiler: set OpenMP/SFML flags manually if needed.")
endif()
