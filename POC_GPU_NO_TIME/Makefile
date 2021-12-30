CC=nvcc
CFLAGS=-I.
DEPS = embedding_map/embedding_map.h time/timecalculate.h FileRW/csvRW.h
OBJ = baseline.o embedding_map/embedding_map.o

POC_GPU: $(OBJ)
	$(CC) -o POC_GPU $(OBJ)

embedding_map/embedding_map.o: embedding_map/embedding_map.cu embedding_map/embedding_map.h
	$(CC) -c -o embedding_map/embedding_map.o embedding_map/embedding_map.cu

baseline.o:baseline.cu $(DEPS) embedding_map/embedding_map.cu
	nvcc -c -o baseline.o baseline.cu



.PHONY:clean

clean:
	rm -r *.o
	rm embedding_map/embedding_map.o
	rm POC_GPU