BIN=main

BIN_DIR=bin
CMD_DIR=cmd
TST_DIR=tst
SRC_DIR=src
INC_DIR=inc
OBJ_DIR=obj

MJKEY=~/.mujoco/mjkey.txt
MUJOCO_PATH=/home/mahan/.mujoco/mujoco200_linux
MJ_FLAGS=-I$(MUJOCO_PATH)/include -L$(MUJOCO_PATH)/bin

CC=clang++
CFLAGS=-I. -I$(INC_DIR)/ $(MJ_FLAGS) -std=c++17 -O3 -pthread -mavx -Wl,-rpath,'$$ORIGIN'

LIBS=-lpthread -fopenmp
LIBS_GL=-lmujoco200 -lGLEW -lGLU -lGL -lglfw
LIBS_NOGL=-lmujoco200nogl

SRCS=$(wildcard $(SRC_DIR)/*.c*) $(wildcard $(SRC_DIR)/*/*.c*)
OBJS=$(addprefix $(OBJ_DIR)/, $(patsubst $(SRC_DIR)/%.cpp, %.o, $(SRCS)))

OUT_DIRS=$(sort $(dir $(OBJS)))


.PHONY: prebuild clean

basic: prebuild $(OUT_DIRS) $(OBJS)
	$(CC) $(CFLAGS) $(LIBS) $(LIBS_GL) -o $(BIN_DIR)/base $(SRCS) $(CMD_DIR)/basic.cpp

tests: prebuild $(OUT_DIRS) $(OBJS)
	$(CC) $(CFLAGS) $(LIBS) $(LIBS_GL) -o $(BIN_DIR)/test_derivatives $(SRCS) $(TST_DIR)/test_derivatives.cpp

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) -c -g $(CFLAGS) -o $@ $<

$(OUT_DIRS):
	mkdir -p $(OUT_DIRS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) mjkey.txt

prebuild:
	cp $(MJKEY) .
	mkdir -p bin
	cp $(MUJOCO_PATH)/bin/libmujoco* ./bin/
