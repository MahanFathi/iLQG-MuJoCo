cmake_minimum_required(VERSION 3.9)
project(ilgq)

set(DIR /home/mahan/Mahan/UnderConstruction/ilgq)


#include directories
INCLUDE_DIRECTORIES(./include)
INCLUDE_DIRECTORIES(./bin)
LINK_DIRECTORIES(./bin)

set(CMAKE_CXX_STANDARD 11)

#set(SOURCES DIR/sample/basic.cpp)

add_executable(ilgq ./main.cpp
                    ./src/ilqr.cpp
                    ./src/test.cpp
#                    ./sample/basic.cpp
#                    ./sample/derivative.cpp
                    ./include/ilqr.h
                    ./include/mujoco.h
                    ./include/mjdata.h
                    ./include/mjmodel.h
                    ./include/mjrender.h
                    ./include/mjvisualize.h
                    ./include/mjxmacro.h
                    ./include/glfw3.h src/lindy.cpp include/lindy.h src/ilqr.cpp include/ilqr.h src/cost.cpp include/cost.h src/cost_derivatives.cpp src/anim.cpp include/anim.h include/types.h config_inv_pen.h)