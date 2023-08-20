CC = icc
CFLAGS = -qopenmp -std=c++17 -O2 -DTBB
MKL_LIBS = -L"/hzy/gyl/intel/oneapi/mkl/2022.2.0/lib/intel64" -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ltbbmalloc
MKL_INCS = -I/hzy/gyl/intel/oneapi/tbb/2021.7.0/include/tbb -I/hzy/gyl/intel/oneapi/mkl/2023.0.0/include -I./inc

all: mkl_spgemm cblas_dxxmm

mkl_spgemm: obj/mkl_spgemm.o obj/Timings.o
	$(CC) -o mkl_spgemm obj/mkl_spgemm.o obj/Timings.o $(CFLAGS) $(MKL_LIBS) $(MKL_INCS)

cblas_dxxmm: cblas_dxxmm.o
	$(CC) -o cblas_dxxmm cblas_dxxmm.o -lmkl_rt $(MKL_INCS)

obj/mkl_spgemm.o: src/mkl_spgemm.cpp inc/Timings.h
	$(CC) -c -o obj/mkl_spgemm.o src/mkl_spgemm.cpp $(CFLAGS) $(MKL_INCS)

obj/Timings.o: src/Timings.cpp inc/Timings.h
	$(CC) -c -o obj/Timings.o src/Timings.cpp $(CFLAGS) $(MKL_INCS)

cblas_dxxmm.o: src/cblas_dxxmm.c
	$(CC) -c -o cblas_dxxmm.o src/cblas_dxxmm.c $(MKL_INCS)

clean:
	rm -f obj/*.o mkl_spgemm cblas_dxxmm
