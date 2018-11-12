dune_add_test(NAME RosiRichards2cDiff
              SOURCES RosiRichards2cDiff.cc
              CMAKE_GUARD "( dune-foamgrid_FOUND AND HAVE_UG )")
dune_symlink_to_source_files(FILES "RosiRichards2cDiff.input")
set(CMAKE_BUILD_TYPE Release)
