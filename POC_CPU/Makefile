CC=nvcc
CFLAGS=-I.
DEPS = embedding_map.h timecalculate.h csvRW.h
OBJ = baseline.o
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

POC_CPU: $(OBJ)
	$(CC)	 -o $@ $^ $(CFLAGS)
clean:
	rm *.o
	rm POC_CPU