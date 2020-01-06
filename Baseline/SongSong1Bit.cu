#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

__device__  int plus_scan(int *x)
{
    int i = threadIdx.x; // id of thread executing this instance
    int n = blockDim.x;  // total number of threads in this block
    int offset;          // distance between elements to be added

    for( offset = 1; offset < n; offset *= 2) {
        int t;

        if ( i >= offset ) 
            t = x[i-offset];
        
        __syncthreads();

        if ( i >= offset ) 
            x[i] = t + x[i]; 

        __syncthreads();
    }
    return x[i];
}

__device__ void partition_by_bit(int *values, int bit)
{
    int thread = threadIdx.x;
    int size = blockDim.x;
    int x_i = values[thread];          
    int p_i = (x_i >> bit) & 1;   

    values[thread] = p_i;  

    __syncthreads();

    int T_before = plus_scan(values);
    int T_total  = values[size-1];

    int F_total  = size - T_total;
    __syncthreads();
    if ( p_i )
        values[T_before-1 + F_total] = x_i;
    else
        values[thread - T_before] = x_i;
}

__device__ void radix_sort(int *values)
{
    int  bit;
    for( bit = 0; bit < 32; ++bit )
    {
        partition_by_bit(values, bit);
        __syncthreads();
    }
}

__global__ void sortBlk(int *in, int n, int *out)
{
    extern __shared__ int s[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        s[threadIdx.x] = in[i];
    }
    __syncthreads();
    radix_sort(s);
    __syncthreads();
    if(i < n)
    {
        out[i] =  s[threadIdx.x];
    }
    __syncthreads();
}

void sortParallel(const uint32_t * in, int n, 
    uint32_t * out, 
    int nBits, int * blockSizes)
{
    // TODO
    dim3 blockSize1(blockSizes[0]);
    int *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));

    size_t smemSize = blockSize1.x*sizeof(int);

    CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));
    sortBlk<<<1, blockSize1, smemSize>>>(d_in, n, d_out);
    CHECK(cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
}



// (Partially) Parallel radix sort: implement parallel histogram and parallel scan in counting sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
// Why "int * blockSizes"? 
// Because we may want different block sizes for diffrent kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByDevice(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits, int * blockSizes)
{
    // TODO
	thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

// Radix sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        bool useDevice=false, int * blockSizes=NULL)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix sort parallel 1 bit\n");
        sortParallel(in, n, out, nBits, blockSizes);
    }
    else // use device
    {
    	printf("\nRadix sort by device\n");
        sortByDevice(in, n, out, nBits, blockSizes);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 10);
    //n = 10;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // DETERMINE BLOCK SIZES
    int blockSizes[2] = {1024, 512}; // One for histogram, one for scan
    if (argc == 4)
    {
        blockSizes[0] = atoi(argv[2]);
        blockSizes[1] = atoi(argv[3]);
    }
    printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0], blockSizes[1]);

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    //printArray(in, n);

    // SET UP NBITS
    int nBits = 4; // Default
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", nBits);

    // SORT BY HOST
    sort(in, n, correctOut, nBits, false, blockSizes);
    //printArray(correctOut, n);
    
    // SORT BY DEVICE
    sort(in, n, out, nBits, true, blockSizes);
    checkCorrectness(out, correctOut, n);
    

    // FREE MEMORIES 
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
