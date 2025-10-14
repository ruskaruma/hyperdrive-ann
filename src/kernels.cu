#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cub/cub.cuh>
#include<vector>
#include<algorithm>

__device__ __forceinline__ float warpReduce(float val)
{
    #pragma unroll
    for(int offset=16;offset>0;offset/=2)
    {
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduce(float val)
{
    __shared__ float shared[32];
    int lane=threadIdx.x%32;
    int wid=threadIdx.x/32;
    
    val=warpReduce(val);
    
    if(lane==0)
    {
        shared[wid]=val;
    }
    __syncthreads();
    
    if(wid==0)
    {
        val=(threadIdx.x<blockDim.x/32)?shared[lane]:0.0f;
        val=warpReduce(val);
    }
    
    return val;
}

__global__ void computeSimilarities(const float* __restrict__ db,
                                   const float* __restrict__ queries,
                                   float* __restrict__ similarities,
                                   int dbCount,
                                   int queryCount,
                                   int dim,
                                   int tileSize)
{
    extern __shared__ float sharedMem[];
    
    int queryIdx=blockIdx.x;
    int dbTile=blockIdx.y;
    int tid=threadIdx.x;
    
    if(queryIdx>=queryCount) return;
    
    float* queryTile=&sharedMem[0];
    float* dbTileData=&sharedMem[tileSize];
    
    // Load query vector to shared memory
    for(int i=tid;i<tileSize&&i<dim;i+=blockDim.x)
    {
        queryTile[i]=queries[queryIdx*dim+i];
    }
    
    float sum=0.0f;
    int dbStart=dbTile*tileSize;
    int dbEnd=min(dbStart+tileSize,dbCount);
    
    // Process database vectors in this tile
    for(int dbIdx=dbStart;dbIdx<dbEnd;dbIdx++)
    {
        // Load database vector to shared memory
        for(int i=tid;i<tileSize&&i<dim;i+=blockDim.x)
        {
            dbTileData[i]=db[dbIdx*dim+i];
        }
        __syncthreads();
        
        // Compute dot product using shared memory
        float dot=0.0f;
        for(int i=tid;i<dim;i+=blockDim.x)
        {
            dot+=queryTile[i]*dbTileData[i];
        }
        __syncthreads();
        
        // Reduce across block
        dot=blockReduce(dot);
        
        // Store result
        if(tid==0)
        {
            similarities[queryIdx*dbCount+dbIdx]=dot;
        }
    }
}

__global__ void findTopK(const float* __restrict__ similarities,
                        int* __restrict__ indices,
                        int* __restrict__ topK,
                        int dbCount,
                        int queryCount,
                        int k)
{
    int queryIdx=blockIdx.x;
    int tid=threadIdx.x;
    
    if(queryIdx>=queryCount) return;
    
    extern __shared__ float sharedSim[];
    float* sharedIdx=(float*)(sharedSim+dbCount);
    
    // Copy similarities to shared memory for this query
    for(int i=tid;i<dbCount;i+=blockDim.x)
    {
        sharedSim[i]=similarities[queryIdx*dbCount+i];
        sharedIdx[i]=i;
    }
    __syncthreads();
    
    // Simple selection sort for top-k
    for(int i=0;i<k;i++)
    {
        int maxIdx=i;
        float maxVal=sharedSim[i];
        
        // Find maximum in remaining elements
        for(int j=i+1;j<dbCount;j++)
        {
            if(sharedSim[j]>maxVal)
            {
                maxVal=sharedSim[j];
                maxIdx=j;
            }
        }
        
        // Swap elements
        if(maxIdx!=i)
        {
            float tempSim=sharedSim[i];
            float tempIdx=sharedIdx[i];
            sharedSim[i]=sharedSim[maxIdx];
            sharedIdx[i]=sharedIdx[maxIdx];
            sharedSim[maxIdx]=tempSim;
            sharedIdx[maxIdx]=tempIdx;
        }
        
        // Store result
        if(tid==0)
        {
            topK[queryIdx*k+i]=(int)sharedIdx[i];
        }
    }
}

__global__ void batchTopK(const float* __restrict__ similarities,
                         int* __restrict__ topK,
                         int dbCount,
                         int queryCount,
                         int k)
{
    int queryIdx=blockIdx.x;
    int tid=threadIdx.x;
    
    if(queryIdx>=queryCount) return;
    
    // Use shared memory for local sorting
    extern __shared__ float shared[];
    float* sims=&shared[0];
    int* indices=(int*)&shared[dbCount];
    
    // Initialize indices
    for(int i=tid;i<dbCount;i+=blockDim.x)
    {
        indices[i]=i;
    }
    __syncthreads();
    
    // Copy similarities
    for(int i=tid;i<dbCount;i+=blockDim.x)
    {
        sims[i]=similarities[queryIdx*dbCount+i];
    }
    __syncthreads();
    
    // Partial sort using selection sort
    for(int i=0;i<k;i++)
    {
        int maxIdx=i;
        float maxVal=sims[i];
        
        // Find max in [i, dbCount)
        for(int j=i+1;j<dbCount;j++)
        {
            if(sims[j]>maxVal)
            {
                maxVal=sims[j];
                maxIdx=j;
            }
        }
        
        // Swap
        if(maxIdx!=i)
        {
            float tempSim=sims[i];
            int tempIdx=indices[i];
            sims[i]=sims[maxIdx];
            indices[i]=indices[maxIdx];
            sims[maxIdx]=tempSim;
            indices[maxIdx]=tempIdx;
        }
    }
    
    // Store top-k results
    for(int i=tid;i<k;i+=blockDim.x)
    {
        topK[queryIdx*k+i]=indices[i];
    }
}

void launchSimilarityKernel(const float* db,
                           const float* queries,
                           float* similarities,
                           int dbCount,
                           int queryCount,
                           int dim,
                           int tileSize,
                           cudaStream_t stream)
{
    dim3 grid(queryCount,(dbCount+tileSize-1)/tileSize);
    dim3 block(min(256,dim));
    
    size_t sharedMemSize=(tileSize+dim)*sizeof(float);
    
    computeSimilarities<<<grid,block,sharedMemSize,stream>>>(
        db,queries,similarities,dbCount,queryCount,dim,tileSize);
}

void launchTopKKernel(const float* similarities,
                     int* topK,
                     int dbCount,
                     int queryCount,
                     int k,
                     cudaStream_t stream)
{
    dim3 grid(queryCount);
    dim3 block(min(256,dbCount));
    
    size_t sharedMemSize=(dbCount*sizeof(float)+dbCount*sizeof(int));
    
    batchTopK<<<grid,block,sharedMemSize,stream>>>(
        similarities,topK,dbCount,queryCount,k);
}
