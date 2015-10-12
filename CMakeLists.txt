project(PoissonRecon)
cmake_minimum_required(VERSION 2.8)

option(USE_OPENMP "use openmp")

if(USE_OPENMP)
	add_definitions(-D_OPENMP)
	if (MSVC)
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    endif(MSVC)
endif(USE_OPENMP)

include_directories (include Src)
add_subdirectory (Src)

install (DIRECTORY include DESTINATION "." FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
