all:
	nvcc Source.cu -lm -o a.out && ./a.out && rm a.out