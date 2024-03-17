all:
	# nvcc Source.cu -o a.out && ./a.out && rm a.out
	nvcc test.cu -o a.out && ./a.out && rm a.out