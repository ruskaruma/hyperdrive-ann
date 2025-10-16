#include<cuda_runtime.h>
#include<cub/cub.cuh>
#include<iostream>

__global__ void computeSimilaritiesKernel(const float* db,const float* queries,float* similarities,int dbCount,int queryCount,int dim,int tileSize)
{
    int queryIdx=blockIdx.x;
    int dbIdx=threadIdx.x+blockIdx.y*blockDim.x;
    if(queryIdx>=queryCount||dbIdx>=dbCount) return;
    
    const float* query=&queries[queryIdx*dim];
    const float* dbVector=&db[dbIdx*dim];
    
    float dot=0.0f;
    for(int i=0;i<dim;i++)
    {
        dot+=query[i]*dbVector[i];
    }
    
    similarities[queryIdx*dbCount+dbIdx]=dot;
}

__global__ void findTopKKernel(const float* similarities,int* topK,int dbCount,int queryCount,int k)
{
    int queryIdx=blockIdx.x;
    if(queryIdx>=queryCount) return;
    
    __shared__ float sharedSims[1024];
    __shared__ int sharedIndices[1024];
    
    int tid=threadIdx.x;
    int blockSize=blockDim.x;
    
    // Initialize with worst possible values
    for(int i=tid;i<k;i+=blockSize)
    {
        sharedSims[i]=-2.0f;
        sharedIndices[i]=-1;
    }
    __syncthreads();
    
    // Process database in chunks
    for(int start=0;start<dbCount;start+=blockSize)
    {
        int dbIdx=start+tid;
        float sim=-2.0f;
        int idx=-1;
        
        if(dbIdx<dbCount)
        {
            sim=similarities[queryIdx*dbCount+dbIdx];
            idx=dbIdx;
        }
        __syncthreads();
        
        // Insert into top-k if better than current worst
        if(sim>sharedSims[k-1])
        {
            // Find insertion point
            int insertPos=k;
            for(int i=0;i<k;i++)
            {
                if(sim>sharedSims[i])
                {
                    insertPos=i;
                    break;
                }
            }
            
            // Shift elements and insert
            if(insertPos<k)
            {
                for(int i=k-1;i>insertPos;i--)
                {
                    sharedSims[i]=sharedSims[i-1];
                    sharedIndices[i]=sharedIndices[i-1];
                }
                sharedSims[insertPos]=sim;
                sharedIndices[insertPos]=idx;
            }
        }
        __syncthreads();
    }
    
    // Copy results
    if(tid<k)
    {
        topK[queryIdx*k+tid]=sharedIndices[tid];
    }
}

void launchSimilarityKernel(const float* db,const float* queries,float* similarities,int dbCount,int queryCount,int dim,int tileSize,cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(queryCount,(dbCount+block.x-1)/block.x);
    computeSimilaritiesKernel<<<grid,block,0,stream>>>(db,queries,similarities,dbCount,queryCount,dim,tileSize);
}

void launchTopKKernel(const float* similarities,int* topK,int dbCount,int queryCount,int k,cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(queryCount);
    findTopKKernel<<<grid,block,0,stream>>>(similarities,topK,dbCount,queryCount,k);
}