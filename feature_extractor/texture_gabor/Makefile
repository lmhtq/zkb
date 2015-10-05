CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = texture_gabor.o

default:texture_gabor

all:texture_gabor

texture_gabor.o:texture_gabor.cpp
	$(CC) -c texture_gabor.cpp

texture_gabor: texture_gabor.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/texture_gabor texture_gabor.o $(POSTFLAGS)

clean:
	rm $(objs)
