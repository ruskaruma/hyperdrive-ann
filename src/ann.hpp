#pragma once

#include<vector>
#include<cuda_runtime.h>

class HyperdriveANN
{
private:
    float* db_device_;
    float* queries_device_;
    float* similarities_device_;
    int* topk_device_;
    size_t db_size_;
    size_t query_size_;
    size_t similarity_size_;
    size_t topk_size_;
    int db_count_;
    int query_count_;
    int dim_;
    int k_;
    int tile_size_;
    cudaStream_t stream_;
    bool initialized_;
    
public:
    HyperdriveANN();
    ~HyperdriveANN();
    
    bool initialize(int dbCount,int dim,int queryCount,int k,int tileSize=256);
    void cleanup();
    
    std::vector<std::vector<int>> topK(const float* db,int dbN,int dim,const float* queries,int qN,int k);
    
    void setTileSize(int tileSize);
    int getTileSize() const;
    
    bool isInitialized() const;
    
private:
    bool allocateMemory();
    void freeMemory();
    void computeSimilarities();
    void findTopK();
    void copyResults(std::vector<std::vector<int>>& results);
};

// Convenience function for simple usage
std::vector<std::vector<int>> topK(const float* db,size_t dbN,int dim,const float* queries,size_t qN,int k);
