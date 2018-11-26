set (MODULE_NAME poisson_recon)

# Get all compilation units.
file (GLOB ${MODULE_NAME}_SRCS *.cpp)

# Create the library.
add_library (${MODULE_NAME} STATIC ${${MODULE_NAME}_SRCS})

install (TARGETS ${MODULE_NAME}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )