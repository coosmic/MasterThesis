# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/solomon/Thesis/MasterThesis/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/solomon/Thesis/MasterThesis/build

# Include any dependencies generated for this target.
include CMakeFiles/pgm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pgm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pgm.dir/flags.make

CMakeFiles/pgm.dir/main.cpp.o: CMakeFiles/pgm.dir/flags.make
CMakeFiles/pgm.dir/main.cpp.o: /home/solomon/Thesis/MasterThesis/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/solomon/Thesis/MasterThesis/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pgm.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pgm.dir/main.cpp.o -c /home/solomon/Thesis/MasterThesis/src/main.cpp

CMakeFiles/pgm.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pgm.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/solomon/Thesis/MasterThesis/src/main.cpp > CMakeFiles/pgm.dir/main.cpp.i

CMakeFiles/pgm.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pgm.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/solomon/Thesis/MasterThesis/src/main.cpp -o CMakeFiles/pgm.dir/main.cpp.s

# Object files for target pgm
pgm_OBJECTS = \
"CMakeFiles/pgm.dir/main.cpp.o"

# External object files for target pgm
pgm_EXTERNAL_OBJECTS =

pgm: CMakeFiles/pgm.dir/main.cpp.o
pgm: CMakeFiles/pgm.dir/build.make
pgm: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_people.so
pgm: /usr/lib/x86_64-linux-gnu/libboost_system.so
pgm: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
pgm: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
pgm: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
pgm: /usr/lib/x86_64-linux-gnu/libboost_regex.so
pgm: /usr/lib/x86_64-linux-gnu/libqhull.so
pgm: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libfreetype.so
pgm: /usr/lib/x86_64-linux-gnu/libz.so
pgm: /usr/lib/x86_64-linux-gnu/libjpeg.so
pgm: /usr/lib/x86_64-linux-gnu/libpng.so
pgm: /usr/lib/x86_64-linux-gnu/libtiff.so
pgm: /usr/lib/x86_64-linux-gnu/libexpat.so
pgm: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_features.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_search.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_io.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
pgm: /usr/lib/x86_64-linux-gnu/libpcl_common.so
pgm: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libfreetype.so
pgm: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
pgm: /usr/lib/x86_64-linux-gnu/libz.so
pgm: /usr/lib/x86_64-linux-gnu/libGLEW.so
pgm: /usr/lib/x86_64-linux-gnu/libSM.so
pgm: /usr/lib/x86_64-linux-gnu/libICE.so
pgm: /usr/lib/x86_64-linux-gnu/libX11.so
pgm: /usr/lib/x86_64-linux-gnu/libXext.so
pgm: /usr/lib/x86_64-linux-gnu/libXt.so
pgm: CMakeFiles/pgm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/solomon/Thesis/MasterThesis/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pgm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pgm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pgm.dir/build: pgm

.PHONY : CMakeFiles/pgm.dir/build

CMakeFiles/pgm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pgm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pgm.dir/clean

CMakeFiles/pgm.dir/depend:
	cd /home/solomon/Thesis/MasterThesis/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/solomon/Thesis/MasterThesis/src /home/solomon/Thesis/MasterThesis/src /home/solomon/Thesis/MasterThesis/build /home/solomon/Thesis/MasterThesis/build /home/solomon/Thesis/MasterThesis/build/CMakeFiles/pgm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pgm.dir/depend

