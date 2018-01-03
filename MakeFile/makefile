COMMON=-O2 -I../include -I../src  -L../bin -std=c++11 -mavx

default:
#	g++ $(COMMON) test.cpp -lmujoco150nogl -o ../bin/test
#	g++ $(COMMON) compile.cpp -lmujoco150nogl -o ../bin/compile
#	g++ $(COMMON) derivative.cpp -lmujoco150nogl -fopenmp -o ../bin/derivative
#	g++ $(COMMON) simulate.cpp -lmujoco150 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/simulate
#	g++ $(COMMON) record.cpp -lmujoco150 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/record
#	g++ $(COMMON) -I/usr/local/include/eigen3 basic.cpp -lmujoco150 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/basic
	g++ $(COMMON) -I/usr/local/include/eigen3 ../main.cpp ../src/lindy.cpp ../src/ilqr.cpp ../src/cost.cpp -lmujoco150 -fopenmp -lGL -lglew ../bin/libglfw.so.3  -o ../bin/main
	g++ $(COMMON) -I/usr/local/include/eigen3 ../src/test.cpp ../src/lindy.cpp -lmujoco150nogl -fopenmp -o ../bin/test
		
egl:	
	g++ $(COMMON) -DMJ_EGL record.cpp -lmujoco150 -lOpenGL -lEGL -lglewegl -o ../bin/recordegl

osmesa:	
	g++ $(COMMON) -DMJ_OSMESA record.cpp -lmujoco150 -lOSMesa -lglewosmesa -o ../bin/recordosmesa

all: default egl osmesa
