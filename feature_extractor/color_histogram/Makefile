CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = color_histogram.o

default:color_histogram

all:color_histogram

color_histogram.o:color_histogram.cpp
	$(CC) -c color_histogram.cpp

color_histogram: color_histogram.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/color_histogram color_histogram.o $(POSTFLAGS)

clean:
	rm $(objs)
