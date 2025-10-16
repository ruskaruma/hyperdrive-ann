#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include"../src/ann.hpp"

void generateTestData(std::vector<float>& db,std::vector<float>& queries,int dbCount,int queryCount,int dim)
{
    db.resize(dbCount*dim);
    queries.resize(queryCount*dim);
    
    std::srand(42); // Fixed seed for reproducibility
    
    // Generate database vectors
    for(int i=0;i<dbCount*dim;i++)
    {
        db[i]=(float)std::rand()/RAND_MAX*2.0f-1.0f;
    }
    
    // Generate query vectors
    for(int i=0;i<queryCount*dim;i++)
    {
        queries[i]=(float)std::rand()/RAND_MAX*2.0f-1.0f;
    }
    
    // Normalize all vectors
    for(int i=0;i<dbCount;i++)
    {
        float* vec=&db[i*dim];
        float norm=0.0f;
        for(int j=0;j<dim;j++)
        {
            norm+=vec[j]*vec[j];
        }
        norm=1.0f/std::sqrt(norm);
        for(int j=0;j<dim;j++)
        {
            vec[j]*=norm;
        }
    }
    
    for(int i=0;i<queryCount;i++)
    {
        float* vec=&queries[i*dim];
        float norm=0.0f;
        for(int j=0;j<dim;j++)
        {
            norm+=vec[j]*vec[j];
        }
        norm=1.0f/std::sqrt(norm);
        for(int j=0;j<dim;j++)
        {
            vec[j]*=norm;
        }
    }
}

float cosineSimilarity(const float* a,const float* b,int dim)
{
    float dot=0.0f;
    for(int i=0;i<dim;i++)
    {
        dot+=a[i]*b[i];
    }
    return dot;
}

std::vector<int> topKCPU(const float* db,int dbCount,int dim,const float* query,int k)
{
    std::vector<std::pair<float,int>> scores;
    scores.reserve(dbCount);
    
    for(int i=0;i<dbCount;i++)
    {
        float sim=cosineSimilarity(query,&db[i*dim],dim);
        scores.emplace_back(sim,i);
    }
    
    std::nth_element(scores.begin(),scores.begin()+k,scores.end(),
        [](const auto& a,const auto& b){return a.first>b.first;});
    
    std::sort(scores.begin(),scores.begin()+k,
        [](const auto& a,const auto& b){return a.first>b.first;});
    
    std::vector<int> result;
    result.reserve(k);
    for(int i=0;i<k;i++)
    {
        result.push_back(scores[i].second);
    }
    
    return result;
}

bool compareResults(const std::vector<std::vector<int>>& gpuResults,const std::vector<int>& cpuResults,int k)
{
    if(gpuResults.size()!=1)
    {
        std::cout<<"ERROR: Expected 1 query result, got "<<gpuResults.size()<<"\n";
        return false;
    }
    
    if(gpuResults[0].size()!=k)
    {
        std::cout<<"ERROR: Expected "<<k<<" results, got "<<gpuResults[0].size()<<"\n";
        return false;
    }
    
    if(cpuResults.size()!=k)
    {
        std::cout<<"ERROR: CPU expected "<<k<<" results, got "<<cpuResults.size()<<"\n";
        return false;
    }
    
    for(int i=0;i<k;i++)
    {
        if(gpuResults[0][i]!=cpuResults[i])
        {
            std::cout<<"ERROR: Mismatch at position "<<i<<": GPU="<<gpuResults[0][i]<<", CPU="<<cpuResults[i]<<"\n";
            return false;
        }
    }
    
    return true;
}

bool testBasicFunctionality()
{
    std::cout<<"Testing basic functionality...\n";
    
    const int dbCount=1000;
    const int queryCount=1;
    const int dim=64;
    const int k=5;
    
    std::vector<float> db,queries;
    generateTestData(db,queries,dbCount,queryCount,dim);
    
    // Test GPU implementation
    HyperdriveANN ann;
    if(!ann.initialize(dbCount,dim,queryCount,k))
    {
        std::cout<<"ERROR: Failed to initialize GPU ANN\n";
        return false;
    }
    
    auto gpuResults=ann.topK(db.data(),dbCount,dim,queries.data(),queryCount,k);
    
    // Test CPU baseline
    auto cpuResults=topKCPU(db.data(),dbCount,dim,queries.data(),k);
    
    // Compare results
    if(!compareResults(gpuResults,cpuResults,k))
    {
        std::cout<<"FAILED: GPU and CPU results don't match\n";
        return false;
    }
    
    std::cout<<"PASSED: Basic functionality test\n";
    return true;
}

bool testMultipleQueries()
{
    std::cout<<"Testing multiple queries...\n";
    
    const int dbCount=500;
    const int queryCount=3;
    const int dim=32;
    const int k=3;
    
    std::vector<float> db,queries;
    generateTestData(db,queries,dbCount,queryCount,dim);
    
    // Test GPU implementation
    HyperdriveANN ann;
    if(!ann.initialize(dbCount,dim,queryCount,k))
    {
        std::cout<<"ERROR: Failed to initialize GPU ANN\n";
        return false;
    }
    
    auto gpuResults=ann.topK(db.data(),dbCount,dim,queries.data(),queryCount,k);
    
    if(gpuResults.size()!=queryCount)
    {
        std::cout<<"ERROR: Expected "<<queryCount<<" query results, got "<<gpuResults.size()<<"\n";
        return false;
    }
    
    // Test each query against CPU
    for(int q=0;q<queryCount;q++)
    {
        auto cpuResults=topKCPU(db.data(),dbCount,dim,&queries[q*dim],k);
        
        if(gpuResults[q].size()!=k)
        {
            std::cout<<"ERROR: Query "<<q<<" expected "<<k<<" results, got "<<gpuResults[q].size()<<"\n";
            return false;
        }
        
        for(int i=0;i<k;i++)
        {
            if(gpuResults[q][i]!=cpuResults[i])
            {
                std::cout<<"ERROR: Query "<<q<<" mismatch at position "<<i<<": GPU="<<gpuResults[q][i]<<", CPU="<<cpuResults[i]<<"\n";
                return false;
            }
        }
    }
    
    std::cout<<"PASSED: Multiple queries test\n";
    return true;
}

bool testConvenienceFunction()
{
    std::cout<<"Testing convenience function...\n";
    
    const int dbCount=100;
    const int queryCount=1;
    const int dim=16;
    const int k=3;
    
    std::vector<float> db,queries;
    generateTestData(db,queries,dbCount,queryCount,dim);
    
    // Test convenience function
    auto gpuResults=topK(db.data(),dbCount,dim,queries.data(),queryCount,k);
    
    if(gpuResults.size()!=queryCount)
    {
        std::cout<<"ERROR: Expected "<<queryCount<<" query results, got "<<gpuResults.size()<<"\n";
        return false;
    }
    
    if(gpuResults[0].size()!=k)
    {
        std::cout<<"ERROR: Expected "<<k<<" results, got "<<gpuResults[0].size()<<"\n";
        return false;
    }
    
    std::cout<<"PASSED: Convenience function test\n";
    return true;
}

int main()
{
    std::cout<<"=== HYPERDRIVE ANN TESTS ===\n\n";
    
    bool allPassed=true;
    
    allPassed&=testBasicFunctionality();
    allPassed&=testMultipleQueries();
    allPassed&=testConvenienceFunction();
    
    std::cout<<"\n=== TEST SUMMARY ===\n";
    if(allPassed)
    {
        std::cout<<"ALL TESTS PASSED\n";
        return 0;
    }
    else
    {
        std::cout<<"SOME TESTS FAILED\n";
        return 1;
    }
}
