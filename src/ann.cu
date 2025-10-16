#include"ann.hpp"
#include<cuda_runtime.h>
#include<iostream>
#include<algorithm>
#include<cstring>

// Forward declarations for kernel functions
void launchSimilarityKernel(const float* db,const float* queries,float* similarities,
                           int dbCount,int queryCount,int dim,int tileSize,cudaStream_t stream);
void launchTopKKernel(const float* similarities,int* topK,int dbCount,int queryCount,int k,cudaStream_t stream);

HyperdriveANN::HyperdriveANN():db_device_(nullptr),queries_device_(nullptr),
                               similarities_device_(nullptr),topk_device_(nullptr),
                               db_size_(0),query_size_(0),similarity_size_(0),topk_size_(0),
                               db_count_(0),query_count_(0),dim_(0),k_(0),tile_size_(256),
                               stream_(0),initialized_(false)
{
    cudaStreamCreate(&stream_);
}

HyperdriveANN::~HyperdriveANN()
{
    cleanup();
    if(stream_)
    {
        cudaStreamDestroy(stream_);
    }
}

bool HyperdriveANN::initialize(int dbCount,int dim,int queryCount,int k,int tileSize)
{
    if(initialized_)
    {
        cleanup();
    }
    
    db_count_=dbCount;
    dim_=dim;
    query_count_=queryCount;
    k_=k;
    tile_size_=tileSize;
    
    // Calculate memory sizes
    db_size_=db_count_*dim_*sizeof(float);
    query_size_=query_count_*dim_*sizeof(float);
    similarity_size_=query_count_*db_count_*sizeof(float);
    topk_size_=query_count_*k_*sizeof(int);
    
    if(!allocateMemory())
    {
        return false;
    }
    
    initialized_=true;
    return true;
}

void HyperdriveANN::cleanup()
{
    if(initialized_)
    {
        freeMemory();
        initialized_=false;
    }
}

bool HyperdriveANN::allocateMemory()
{
    cudaError_t err;
    
    err=cudaMalloc(&db_device_,db_size_);
    if(err!=cudaSuccess)
    {
        std::cerr<<"Failed to allocate database memory: "<<cudaGetErrorString(err)<<"\n";
        return false;
    }
    
    err=cudaMalloc(&queries_device_,query_size_);
    if(err!=cudaSuccess)
    {
        std::cerr<<"Failed to allocate queries memory: "<<cudaGetErrorString(err)<<"\n";
        freeMemory();
        return false;
    }
    
    err=cudaMalloc(&similarities_device_,similarity_size_);
    if(err!=cudaSuccess)
    {
        std::cerr<<"Failed to allocate similarities memory: "<<cudaGetErrorString(err)<<"\n";
        freeMemory();
        return false;
    }
    
    err=cudaMalloc(&topk_device_,topk_size_);
    if(err!=cudaSuccess)
    {
        std::cerr<<"Failed to allocate topk memory: "<<cudaGetErrorString(err)<<"\n";
        freeMemory();
        return false;
    }
    
    return true;
}

void HyperdriveANN::freeMemory()
{
    if(db_device_)
    {
        cudaFree(db_device_);
        db_device_=nullptr;
    }
    if(queries_device_)
    {
        cudaFree(queries_device_);
        queries_device_=nullptr;
    }
    if(similarities_device_)
    {
        cudaFree(similarities_device_);
        similarities_device_=nullptr;
    }
    if(topk_device_)
    {
        cudaFree(topk_device_);
        topk_device_=nullptr;
    }
}

std::vector<std::vector<int>> HyperdriveANN::topK(const float* db,int dbN,int dim,const float* queries,int qN,int k)
{
    if(!initialized_)
    {
        if(!initialize(dbN,dim,qN,k,tile_size_))
        {
            return {};
        }
    }
    
    // Copy data to device
    cudaMemcpyAsync(db_device_,db,db_size_,cudaMemcpyHostToDevice,stream_);
    cudaMemcpyAsync(queries_device_,queries,query_size_,cudaMemcpyHostToDevice,stream_);
    
    // Compute similarities
    computeSimilarities();
    
    // Find top-k
    findTopK();
    
    // Copy results back
    std::vector<std::vector<int>> results;
    copyResults(results);
    
    return results;
}

void HyperdriveANN::computeSimilarities()
{
    launchSimilarityKernel(db_device_,queries_device_,similarities_device_,
                          db_count_,query_count_,dim_,tile_size_,stream_);
}

void HyperdriveANN::findTopK()
{
    launchTopKKernel(similarities_device_,topk_device_,db_count_,query_count_,k_,stream_);
}

void HyperdriveANN::copyResults(std::vector<std::vector<int>>& results)
{
    std::vector<int> host_topk(query_count_*k_);
    cudaMemcpyAsync(host_topk.data(),topk_device_,topk_size_,cudaMemcpyDeviceToHost,stream_);
    cudaStreamSynchronize(stream_);
    
    results.resize(query_count_);
    for(int q=0;q<query_count_;q++)
    {
        results[q].resize(k_);
        for(int i=0;i<k_;i++)
        {
            results[q][i]=host_topk[q*k_+i];
        }
    }
}

void HyperdriveANN::setTileSize(int tileSize)
{
    tile_size_=tileSize;
}

int HyperdriveANN::getTileSize() const
{
    return tile_size_;
}

bool HyperdriveANN::isInitialized() const
{
    return initialized_;
}

// Convenience function
std::vector<std::vector<int>> topK(const float* db,size_t dbN,int dim,const float* queries,size_t qN,int k)
{
    HyperdriveANN ann;
    if(!ann.initialize(dbN,dim,qN,k))
    {
        return {};
    }
    return ann.topK(db,dbN,dim,queries,qN,k);
}
