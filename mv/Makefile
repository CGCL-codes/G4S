CC = icc
CFLAGS = -O2 -march=native
LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lmkl_rt -L/hzy/gyl/intel/oneapi/tbb/2021.7.0/lib -ltbb -L/hzy/gyl/intel/oneapi/mkl/2023.0.0/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

TARGET = mv
SRCS = mv.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
