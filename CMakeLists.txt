
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Modules)

# boost
find_package(Boost 1.42.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# unit tests
add_subdirectory(UnitTest++)
include_directories(UnitTest++/src)

find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

find_package(Sqlite3 REQUIRED)
include_directories(${SQLITE3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "-std=c++0x")

include_directories(.)

link_libraries(
	${Boost_LIBRARIES}
	${SQLITE3_LIBRARIES}
)

add_executable(igd_sqlite
	main.cpp
	test_data.h
	test_data.cpp
	train.h
	train.cpp
)

target_link_libraries(igd_sqlite
	UnitTest++
)


