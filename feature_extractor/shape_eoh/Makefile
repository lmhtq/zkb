CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = shape_eoh.o

default:shape_eoh

all:shape_eoh

shape_eoh.o:shape_eoh.cpp
	$(CC) -c shape_eoh.cpp

shape_eoh: shape_eoh.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/shape_eoh shape_eoh.o $(POSTFLAGS)

clean:
	rm $(objs)
