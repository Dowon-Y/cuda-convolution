SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS   :=  -O3

CXX = g++
CXXFLAGS = -std=c++11

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS := q1 q2 q3
all:$(EXECS)


#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt *.cpp
	
#######################################################################
q1: q1.cpp
	$(CXX) -o q1 q1.cpp $(CXXFLAGS)

#######################################################################
q2: q2.cu
	${NVCC} -o q2 q2.cu $(LIB) $(OPTIONS)

#######################################################################
q3: q3.cu
	${NVCC} -o q3 q3.cu $(LIB) $(OPTIONS)

#######################################################################