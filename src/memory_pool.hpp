#pragma once

#include<cuda_runtime.h>
#include<memory>
#include<vector>
#include<mutex>
#include<queue>
#include<unordered_map>

struct MemoryBlock
{
    void* ptr_;
    size_t size_;
    bool in_use_;
    cudaStream_t stream_;
    
    MemoryBlock():ptr_(nullptr),size_(0),in_use_(false),stream_(0){}
    MemoryBlock(void* ptr,size_t size):ptr_(ptr),size_(size),in_use_(false),stream_(0){}
};

class MemoryPool
{
private:
    std::queue<std::unique_ptr<MemoryBlock>> free_blocks_;
    std::unordered_map<void*,std::unique_ptr<MemoryBlock>> allocated_blocks_;
    std::mutex mutex_;
    size_t total_size_;
    size_t max_size_;
    bool initialized_;
    
public:
    MemoryPool();
    ~MemoryPool();
    
    bool initialize(size_t max_size_mb);
    void cleanup();
    
    void* allocate(size_t size,cudaStream_t stream=0);
    void deallocate(void* ptr);
    void synchronize();
    
    size_t getTotalSize() const{return total_size_;}
    size_t getMaxSize() const{return max_size_;}
    size_t getUsedSize() const;
    bool isInitialized() const{return initialized_;}
    
private:
    MemoryBlock* findFreeBlock(size_t size);
    void createNewBlock(size_t size);
};
