all:
	nvcc Source.cu -lcublas -o a.out && ./a.out && rm a.out
	# nvcc test3.cu -lcublas -o a.out && ./a.out && rm a.out