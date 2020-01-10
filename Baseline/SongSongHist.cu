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

__global__ void computeHistKernel2(int * src, int n, int * hist, int nBins, int bit)
{
    // TODO
    // Each block computes its local hist using atomic on SMEM
	extern __shared__ int s[];
	for(int i = threadIdx.x; i < nBins; i += blockDim.x)
		s[i] = 0;
	__syncthreads();
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		int bin = (src[i] >> bit) & (nBins -1);
		atomicAdd(&s[bin], 1);
	}
	__syncthreads();
    // Each block adds its local hist to global hist using atomic on GMEM
	for(int i = threadIdx.x; i < nBins; i += blockDim.x)
		atomicAdd(&hist[i], s[i]);
}

// (Partially) Parallel radix sort: implement parallel histogram and parallel scan in counting sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
// Why "int * blockSizes"? 
// Because we may want different block sizes for diffrent kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortParallel(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits, int * blockSizes)
{
    // TODO
	int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    // In each counting sort, we sort data in "src" and write result to "dst"
    // Then, we swap these 2 pointers and go to the next counting sort
    // At first, we assign "src = in" and "dest = out"
    // However, the data pointed by "in" is read-only 
    // --> we create a copy of this data and assign "src" to the address of this copy
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;
	uint32_t * temp;
	
	dim3 blockSize1(blockSizes[0]);
	dim3 blockSize2(blockSizes[1]);
	
    // Allocate device memories
	int  * d_hist, *d_histScan, * d_in;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_hist, nBins * sizeof(int)));
    CHECK(cudaMalloc(&d_histScan, nBins * sizeof(int)));
    
	// Call kernel
	dim3 gridSize1((n - 1) / blockSize1.x + 1);
	dim3 gridSize2((n - 1) / blockSize2.x + 1);
	
	size_t smemSize = nBins*sizeof(int);
    
    int *d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize2.x * sizeof(int)));
    

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bits)
	// In each loop, sort elements according to the current digit 
	// (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: Compute "hist" of the current digit
		CHECK(cudaMemcpy(d_in, src, n * sizeof(int), cudaMemcpyHostToDevice));

		CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));
		
        computeHistKernel2<<<gridSize1, blockSize1, smemSize>>>(d_in, n, d_hist, nBins, bit);
        CHECK(cudaMemcpy(hist, d_hist, nBins * sizeof(int), cudaMemcpyDeviceToHost));

    	// TODO: Scan "hist" (exclusively) and save the result to "histScan"
        histScan[0] = 0;
        for(int i = 1; i < nBins; i++)
            histScan[i] = histScan[i - 1] + hist[i - 1];

    	// TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
		for(int i = 0; i < n; i++)
		{
			int bin = (src[i] >> bit) & (nBins -1);
			dst[histScan[bin]] = src[i];
			histScan[bin]++;
		}
    	
    	// TODO: Swap "src" and "dst"
		temp = src;
		src = dst;
        dst = temp;
    }

    // TODO: Copy result to "out"
    memcpy(out, src, n * sizeof(uint32_t));
    
    // Free memories
    free(hist);
    free(histScan);
    free(originalSrc);

	// Free device memories
    CHECK(cudaFree(d_in));
	CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan))
	CHECK(cudaFree(d_blkSums));
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
    	printf("\nRadix sort parallel scan hist\n");
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
    int n = (1 << 20);
    //n = 10;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    //printArray(in, n);

    // SET UP NBITS
    int nBits = 4; // Default
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", nBits);

    // DETERMINE BLOCK SIZES
    int blockSizes[2] = {512, 512}; // One for histogram, one for scan
    if (argc == 4)
    {
        blockSizes[0] = atoi(argv[2]);
        blockSizes[1] = atoi(argv[3]);
    }
    printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0], blockSizes[1]);

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
