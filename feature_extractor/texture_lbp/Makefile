CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = texture_lbp.o

default:texture_lbp

all:texture_lbp

texture_lbp.o:texture_lbp.cpp
	$(CC) -c texture_lbp.cpp

texture_lbp: texture_lbp.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/texture_lbp texture_lbp.o $(POSTFLAGS)

clean:
	rm $(objs)
