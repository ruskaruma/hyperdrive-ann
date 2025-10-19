#include"memory_pool.hpp"
#include<iostream>

MemoryPool::MemoryPool():total_size_(0),max_size_(0),initialized_(false)
{
}

MemoryPool::~MemoryPool()
{
    cleanup();
}

bool MemoryPool::initialize(size_t max_size_mb)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if(initialized_)
    {
        cleanup();
    }
    
    max_size_=max_size_mb*1024*1024;
    total_size_=0;
    initialized_=true;
    
    return true;
}

void MemoryPool::cleanup()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if(!initialized_)
    {
        return;
    }
    
    for(auto& pair:allocated_blocks_)
    {
        if(pair.second->ptr_)
        {
            cudaFree(pair.second->ptr_);
        }
    }
    
    allocated_blocks_.clear();
    
    while(!free_blocks_.empty())
    {
        free_blocks_.pop();
    }
    
    total_size_=0;
    initialized_=false;
}

void* MemoryPool::allocate(size_t size,cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if(!initialized_)
    {
        return nullptr;
    }
    
    if(total_size_+size>max_size_)
    {
        std::cerr<<"Memory pool exhausted. Requested: "<<size<<" bytes, Available: "<<(max_size_-total_size_)<<" bytes\n";
        return nullptr;
    }
    
    MemoryBlock* block=findFreeBlock(size);
    if(!block)
    {
        createNewBlock(size);
        block=findFreeBlock(size);
    }
    
    if(!block)
    {
        std::cerr<<"Failed to allocate memory block of size: "<<size<<" bytes\n";
        return nullptr;
    }
    
    block->in_use_=true;
    block->stream_=stream;
    total_size_+=block->size_;
    
    return block->ptr_;
}

void MemoryPool::deallocate(void* ptr)
{
    if(!ptr)
    {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it=allocated_blocks_.find(ptr);
    if(it!=allocated_blocks_.end())
    {
        it->second->in_use_=false;
        total_size_-=it->second->size_;
        free_blocks_.push(std::move(it->second));
        allocated_blocks_.erase(it);
    }
}

void MemoryPool::synchronize()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    for(auto& pair:allocated_blocks_)
    {
        if(pair.second->stream_)
        {
            cudaStreamSynchronize(pair.second->stream_);
        }
    }
}

size_t MemoryPool::getUsedSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return total_size_;
}

MemoryBlock* MemoryPool::findFreeBlock(size_t size)
{
    std::queue<std::unique_ptr<MemoryBlock>> temp_queue;
    MemoryBlock* found_block=nullptr;
    
    while(!free_blocks_.empty())
    {
        auto block=std::move(free_blocks_.front());
        free_blocks_.pop();
        
        if(block->size_>=size&&!block->in_use_)
        {
            found_block=block.get();
            allocated_blocks_[block->ptr_]=std::move(block);
            break;
        }
        
        temp_queue.push(std::move(block));
    }
    
    while(!temp_queue.empty())
    {
        free_blocks_.push(std::move(temp_queue.front()));
        temp_queue.pop();
    }
    
    return found_block;
}

void MemoryPool::createNewBlock(size_t size)
{
    void* ptr=nullptr;
    cudaError_t err=cudaMalloc(&ptr,size);
    
    if(err!=cudaSuccess)
    {
        std::cerr<<"Failed to allocate GPU memory: "<<cudaGetErrorString(err)<<"\n";
        return;
    }
    
    auto block=std::make_unique<MemoryBlock>(ptr,size);
    allocated_blocks_[ptr]=std::move(block);
}
