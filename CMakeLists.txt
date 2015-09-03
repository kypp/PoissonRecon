project(PoissonRecon)
cmake_minimum_required(VERSION 2.8)

include_directories (include Src)
add_subdirectory (Src)

install (DIRECTORY include DESTINATION include FILES_MATCHING PATTERN "*.h")