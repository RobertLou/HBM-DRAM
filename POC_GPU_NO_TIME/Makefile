CC=nvcc
CFLAGS=-I.
DEPS = embedding_map/embedding_map.h time/timecalculate.h FileRW/csvRW.h
OBJ = baseline.o

POC_GPU: $(OBJ)
	$(CC) -o POC_GPU $(OBJ)

baseline.o:baseline.cu $(DEPS)
	nvcc -c -o baseline.o baseline.cu

.PHONY:clean

clean:
	rm *.o
	rm POC_GPU