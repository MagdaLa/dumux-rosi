dune_add_test(SOURCES test_rosi.cc)
dune_symlink_to_source_files(FILES "grids" "test_rosi.input")
#ADD_CUSTOM_COMMAND(COMMAND "ln -s $PWD/.. $PWD/src")
