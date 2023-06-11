CC=nvcc
CFLAGS=-I.
DEPS = embedding_map/embedding_map.h ../time/timecalculate.h ../FileRW/csvRW.h
OBJ = baseline.o embedding_map/embedding_map.o

POC_CPU: $(OBJ)
	$(CC) -o POC_CPU $(OBJ)

baseline.o:baseline.cpp $(DEPS)
	$(CC) -c -o baseline.o baseline.cpp

embedding_map/embedding_map.o:embedding_map/embedding_map.cpp embedding_map/embedding_map.h
	$(CC) -c -o embedding_map/embedding_map.o embedding_map/embedding_map.cpp

.PHONY:clean

clean:
	rm -r *.o
	rm embedding_map/embedding_map.o
	rm POC_CPU