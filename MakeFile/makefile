COMMON=-O2 -I../include -I/usr/local/include/eigen3 -I../src  -L../bin -std=c++11 -mavx

default:
	g++ $(COMMON) ../main.cpp ../src/lindy.cpp ../src/ilqr.cpp ../src/cost.cpp -lmujoco150 -fopenmp -lGL -lglew ../bin/libglfw.so.3  -o ../bin/main
	g++ $(COMMON) ../src/test.cpp ../src/lindy.cpp -lmujoco150nogl -fopenmp -o ../bin/test

all: default egl osmesa
