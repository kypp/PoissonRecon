set (MODULE_NAME poisson_recon)

# Get all compilation units.
file (GLOB ${MODULE_NAME}_SRCS *.cpp)

# Create the library.
add_library (${MODULE_NAME} STATIC ${${MODULE_NAME}_SRCS})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
endif()

if(NOT CMAKE_INSTALL_BINDIR)
	set(CMAKE_INSTALL_BINDIR bin)
endif()

if(NOT CMAKE_INSTALL_LIBDIR)
	set(CMAKE_INSTALL_LIBDIR lib)
endif()

install (TARGETS ${MODULE_NAME}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
