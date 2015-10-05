CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = color_correlogram.o

default:color_correlogram

all:color_correlogram

color_correlogram.o:color_correlogram.cpp
	$(CC) -c color_correlogram.cpp

color_correlogram: color_correlogram.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/color_correlogram color_correlogram.o  $(POSTFLAGS)

clean:
	rm $(objs)
