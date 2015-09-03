set (MODULE_NAME poisson_recon)

# Get all compilation units.
file (GLOB ${MODULE_NAME}_SRCS *.cpp)

# Create the library.
add_library (${MODULE_NAME} ${${MODULE_NAME}_SRCS})

install (TARGETS ${MODULE_NAME} 
		ARCHIVE DESTINATION lib 
		LIBRARY DESTINATION lib)