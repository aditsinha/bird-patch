
CXX=g++
CXXFLAGS= -fopenmp -g $(shell pkg-config --cflags OpenEXR) $(shell pkg-config --cflags opencv) -I/home/adit/Code/mlpack-2.2.0/build/include

LDFLAGS=$(shell pkg-config --libs OpenEXR) $(shell pkg-config --libs opencv) -lrt -larmadillo -L/home/adit/Code/mlpack-2.2.0/build/lib -lmlpack -Wl,-rpath=/home/adit/Code/mlpack-2.2.0/build/lib

OBJS=io.o geometry.o clustering.o

all: main

main: main.cc $(OBJS)
