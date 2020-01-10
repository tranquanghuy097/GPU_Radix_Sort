#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

__device__ void partition_by_bit(int *values, int bit);
__device__  int plus_scan(int *x);
__device__ void radix_sort(int *values);


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
        {
            values[T_before-1 + F_total] = x_i;
        }
        else
        {
            values[thread - T_before] = x_i;
        }
    
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

__global__ void sortBlk(int *in, int n, int *sortedBlocks, int bit, int nBins)
{
    extern __shared__ int s[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        s[threadIdx.x] = (in[i] >> bit) & (nBins - 1);
    }
    __syncthreads();
    radix_sort(s);
    __syncthreads();
    if(i < n)
    {
        sortedBlocks[i] =  s[threadIdx.x];
    }
    __syncthreads();
}

__global__ void computeHistKernel(int * in, int n, int * hist, int nBins, int gridSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		atomicAdd(&hist[blockIdx.x + in[i] * gridSize], 1);
	}
}

__global__ void scanBlkKernel(int * in, int n, int * out)
{   
    //TODO
	extern __shared__ int s[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
        s[threadIdx.x] = in[i];
    else
        s[threadIdx.x] = 0;
	__syncthreads();
	int temp;
	for(int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if(threadIdx.x >= stride)
			temp = s[threadIdx.x - stride];
		__syncthreads();
		if(threadIdx.x >= stride)
			s[threadIdx.x] += temp;
		__syncthreads();
    }
    if(i < n - 1)
        out[i + 1] = s[threadIdx.x];
    out[0] = 0;
}

__global__ void scatterKernel(int * in, int n, int *sortedBlocks, int *histScan, int * out, int gridSize)
{
    extern __shared__ int s[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        s[threadIdx.x] = sortedBlocks[i];
    }
    __syncthreads();
    int before = 0;
    for(int j = threadIdx.x - 1; j >= 0; j--)
        if(s[threadIdx.x] == s[j])
            before++;
    __syncthreads();
    int index = blockIdx.x + sortedBlocks[i] * gridSize;
    int rank = histScan[index] + before;
    out[rank] = in[i];
}

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
void sortBit(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits, int * blockSizes, int bit)
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
	size_t sharedMemorySizeByte = blockSize2.x * sizeof(int);
    
    int *d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize2.x * sizeof(int)));
    

    // TODO: Compute "hist" of the current digit
	CHECK(cudaMemcpy(d_in, src, n * sizeof(int), cudaMemcpyHostToDevice));

	CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));
		
	computeHistKernel2<<<gridSize1, blockSize1, smemSize>>>(d_in, n, d_hist, nBins, bit);

    // TODO: Scan "hist" (exclusively) and save the result to "histScan"
    scanBlkKernel<<<gridSize2, blockSize2, sharedMemorySizeByte>>>(d_hist, nBins, d_histScan);
    CHECK(cudaMemcpy(hist, d_histScan, nBins * sizeof(int), cudaMemcpyDeviceToHost));

    // TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
	for(int i = 0; i < n; i++)
	{
		int bin = (src[i] >> bit) & (nBins -1);
		dst[hist[bin]] = src[i];
		hist[bin]++;
	}
    	
    // TODO: Swap "src" and "dst"
	temp = src;
	src = dst;
    dst = temp;

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

void sortParallel(const uint32_t * in, int n, 
    uint32_t * out, 
    int nBits, int * blockSizes)
{
// TODO
    int nBins = 1 << nBits; // 2^nBits

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * k = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;
    uint32_t * temp;

    dim3 blockSize1(blockSizes[0]);
    dim3 blockSize2(blockSizes[1]);

    // Allocate device memories
    int  * d_hist, *d_histScan, * d_in, *d_sortedBlocks, *d_out, *d_k;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CHECK(cudaMalloc(&d_sortedBlocks, n * sizeof(int)));
    CHECK(cudaMalloc(&d_k, n * sizeof(int)));

    // Call kernel
    dim3 gridSize1((n - 1) / blockSize1.x + 1);
    dim3 gridSize2((n - 1) / blockSize2.x + 1);

    CHECK(cudaMalloc(&d_hist, nBins * gridSize1.x * sizeof(int)));
    CHECK(cudaMalloc(&d_histScan, nBins * gridSize1.x * sizeof(int)));

    int * hist = (int *)malloc(nBins * gridSize1.x * sizeof(int));
    int * histScan = (int *)malloc(nBins * gridSize1.x * sizeof(int));

    size_t smemSize = blockSize1.x*sizeof(int);

    uint32_t *block = (uint32_t *)malloc(blockSize1.x * sizeof(int));
    uint32_t *block2 = (uint32_t *)malloc(blockSize1.x * sizeof(int));
    int m = 0;
    int mul;
    

    GpuTimer timer; 
    int i = 0;
       
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        printf("%d: \n", i);
        timer.Start();
        CHECK(cudaMemcpy(d_in, src, n * sizeof(int), cudaMemcpyHostToDevice));
        sortBlk<<<gridSize1, blockSize1, smemSize>>>(d_in, n, d_sortedBlocks, bit, nBins);
        for(int j = 0; j < n; j++)
        {
            block[m] = src[j];
            m++;
            if((j + 1) % blockSize1.x == 0)
            {
                m = 0;
                sortBit(block, blockSize1.x, block2, nBits, blockSizes, bit);
                mul = (j + 1) / blockSize1.x;
                for(int l = j + 1 - blockSize1.x; l < mul * blockSize1.x; l++)
                {
                    k[l] = block2[m];
                    m++;
                }
                m = 0;
            }
        }
        CHECK(cudaMemcpy(d_k, k, n * sizeof(int), cudaMemcpyHostToDevice));
        timer.Stop();
        printf("Sort block: %.3f ms\n", timer.Elapsed());
       
        // TODO: Compute "hist" of the current digit

        timer.Start();
        CHECK(cudaMemset(d_hist, 0, nBins * gridSize1.x * sizeof(int)));
        computeHistKernel<<<gridSize1, blockSize1>>>(d_sortedBlocks, n, d_hist, nBins, gridSize1.x);
        CHECK(cudaMemcpy(hist, d_hist, nBins * gridSize1.x * sizeof(int), cudaMemcpyDeviceToHost));
        timer.Stop();
        printf("Hist: %.3f ms\n", timer.Elapsed());

        //TODO: Scan "hist" (exclusively) and save the result to "histScan"
        timer.Start();
        histScan[0] = 0;
        for (int bin = 1; bin < nBins * gridSize1.x; bin++)
            histScan[bin] = histScan[bin - 1] + hist[bin - 1];
        CHECK(cudaMemcpy(d_histScan, histScan, nBins * gridSize1.x * sizeof(int), cudaMemcpyHostToDevice));
        timer.Stop();
        printf("Scan: %.3f ms\n", timer.Elapsed());
        
        // TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
        scatterKernel<<<gridSize1, blockSize1, smemSize>>>(d_k, n, d_sortedBlocks, d_histScan, d_out, gridSize1.x);
        CHECK(cudaMemcpy(dst, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
        timer.Stop();
        printf("Scatter: %.3f ms\n", timer.Elapsed());
        
        // TODO: Swap "src" and "dst"
        temp = src;
        src = dst;
        dst = temp;
        i++;
    }

    // TODO: Copy result to "out"
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memories
    free(originalSrc);
    free(block);
    free(block2);

    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
    CHECK(cudaFree(d_sortedBlocks));
    CHECK(cudaFree(d_k));
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
    	printf("\nRadix sort Satish parallel\n");
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
            printf("%d\n", i);
            printf("%d\n", out[i]);
            printf("%d\n", correctOut[i]);
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
    //n = 16384;
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

    sort(in, n, out, nBits, false, blockSizes);
    //printArray(correctOut, n);
    
    // SORT BY DEVICE
    sort(in, n, correctOut, nBits, true, blockSizes);
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES 
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
