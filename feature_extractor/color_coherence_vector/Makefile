CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

objs = color_coherence_vector.o

default:color_coherence_vector

all:color_coherence_vector

color_coherence_vector.o:color_coherence_vector.cpp
	$(CC) -c color_coherence_vector.cpp

color_coherence_vector: color_coherence_vector.o
	$(CC) $(PREFLAGS) -o $(BINSDIR)/color_coherence_vector color_coherence_vector.o $(POSTFLAGS)

clean:
	rm $(objs)
