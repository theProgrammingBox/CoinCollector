all:
	nvcc Source.cu -lm -o a.out && ./a.out && rm a.out
	# gcc test2.c -lm -o a.out && ./a.out && rm a.out