CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = color_moment.o

default:color_moment

all:color_moment

color_moment.o:color_moment.cpp
	$(CC) -c color_moment.cpp

color_moment: color_moment.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/color_moment color_moment.o $(POSTFLAGS)

clean:
	rm $(objs)
