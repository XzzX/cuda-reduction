all: 
	nvcc -O3 -DNDEBUG -o reduction reduction.cu

benchmark:
	nvprof ./reduction
	nvprof --metrics gld_throughput,gld_efficiency,inst_per_warp,shared_load_transactions ./reduction

