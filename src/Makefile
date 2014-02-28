CXX      = g++
CXXFLAGS = -Wall -Werror $(shell pkg-config --cflags libzmq opencv) -O3 -march=native
LDLIBS   = $(shell pkg-config --libs libzmq opencv) -L /usr/local/lib64 -lfreenect_sync -lfreenect_cv -lm
SRCS     = depth-tracking.cc
OBJS     = $(SRCS:.c=.o)

all: color-tracking flow-tracking depth-tracking

color-tracking: $(OBJS)

flow-tracking: flow-tracking.o
depth-tracking: depth-tracking.o
.PHONY: all clean


clean:
	rm -f *.o color-tracking flow-tracking depth-tracking
