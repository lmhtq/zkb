CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = shape_hog.o

default:shape_hog

all:shape_hog

shape_hog.o:shape_hog.cpp
	$(CC) -c shape_hog.cpp

shape_hog: shape_hog.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/shape_hog shape_hog.o $(POSTFLAGS)

clean:
	rm $(objs)
