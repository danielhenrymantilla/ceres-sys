cmake_minimum_required(VERSION 2.8.12)

project(cereswrapper)

find_package(Ceres REQUIRED)

set(TARGET cereswrapper)

set(SRC
	cereswrapper.h
	cereswrapper.c
)

add_library(${TARGET} ${SRC})

install(TARGETS ${TARGET}
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)

INSTALL(FILES
	cereswrapper.h
	DESTINATION ${CMAKE_INSTALL_PREFIX}
)