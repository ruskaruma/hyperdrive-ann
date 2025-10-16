#include<iostream>
#include<vector>
#include<string>
#include<chrono>
#include<cuda_runtime.h>
#include<random>
#include<cmath>
#include<cstring>
#include<iomanip>
#include<sstream>
#include"ann.hpp"

void printBanner()
{
    std::cout<<"\n";
    std::cout<<"  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n";
    std::cout<<"  | H | Y | P | E | R | D | R | I | V | E |   | A | N | N |       |\n";
    std::cout<<"  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n";
    std::cout<<"\n";
    std::cout<<"                    GPU-Accelerated ANN Search Engine\n";
    std::cout<<"                        Powered by CUDA 12.8\n";
    std::cout<<"\n";
}

void printUsage(const char* programName)
{
    printBanner();
    std::cout<<"Usage: "<<programName<<" [options]\n";
    std::cout<<"Options:\n";
    std::cout<<"  --help                    Show this help message\n";
    std::cout<<"  --go                      Quick demo with synthetic data (recommended)\n";
    std::cout<<"  --generate <count> <dim> <filename>  Generate synthetic data\n";
    std::cout<<"  --benchmark <db_file> <query_file> <k>  Run GPU vs CPU benchmark\n";
    std::cout<<"  --tile-size <size>        Set GPU tile size (default: 256)\n";
    std::cout<<"  --iterations <n>          Number of benchmark iterations (default: 1)\n";
    std::cout<<"  --warmup <n>              Number of warmup runs (default: 1)\n";
    std::cout<<"\n";
    std::cout<<"Quick Start:\n";
    std::cout<<"  "<<programName<<" --go\n";
    std::cout<<"\n";
    std::cout<<"Examples:\n";
    std::cout<<"  "<<programName<<" --generate 1000000 128 db.bin\n";
    std::cout<<"  "<<programName<<" --generate 16 128 queries.bin\n";
    std::cout<<"  "<<programName<<" --benchmark db.bin queries.bin 10\n";
}

void generateData(int count,int dim,const std::string& filename)
{
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f,1.0f);
    
    std::vector<float> vectors(count*dim);
    
    // Generate random vectors
    for(int i=0;i<count*dim;i++)
    {
        vectors[i]=dist(gen);
    }
    
    // Normalize vectors
    for(int i=0;i<count;i++)
    {
        float* vec=&vectors[i*dim];
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
    
    // Save to file
    FILE* f=fopen(filename.c_str(),"wb");
    if(!f)
    {
        std::cerr<<"Failed to open "<<filename<<" for writing\n";
        return;
    }
    
    fwrite(&count,sizeof(int),1,f);
    fwrite(&dim,sizeof(int),1,f);
    fwrite(vectors.data(),sizeof(float),count*dim,f);
    fclose(f);
    
    std::cout<<"Generated "<<count<<" vectors of dimension "<<dim<<" to "<<filename<<"\n";
}

void loadData(const std::string& filename,std::vector<float>& vectors,int& count,int& dim)
{
    FILE* f=fopen(filename.c_str(),"rb");
    if(!f)
    {
        std::cerr<<"Failed to open "<<filename<<" for reading\n";
        return;
    }
    
    fread(&count,sizeof(int),1,f);
    fread(&dim,sizeof(int),1,f);
    vectors.resize(count*dim);
    fread(vectors.data(),sizeof(float),count*dim,f);
    fclose(f);
    
    std::cout<<"Loaded "<<count<<" vectors of dimension "<<dim<<" from "<<filename<<"\n";
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

void printDeviceInfo()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    
    std::cout<<"+==============================================================+\n";
    std::cout<<"|                        DEVICE INFO                           |\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"| GPU: "<<std::left<<std::setw(56)<<prop.name<<"|\n";
    std::cout<<"| Compute Capability: "<<std::setw(43)<<prop.major<<"."<<prop.minor<<"|\n";
    std::cout<<"| Memory: "<<std::setw(48)<<prop.totalGlobalMem/1024/1024/1024<<"GB|\n";
    std::cout<<"| Multiprocessors: "<<std::setw(44)<<prop.multiProcessorCount<<"|\n";
    std::cout<<"| Max Threads/Block: "<<std::setw(41)<<prop.maxThreadsPerBlock<<"|\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"\n";
}

void benchmark(int iterations,int warmup,int tileSize)
{
    printBanner();
    printDeviceInfo();
    
    std::cout<<"+==============================================================+\n";
    std::cout<<"|                      BENCHMARK CONFIG                        |\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"| Iterations: "<<std::setw(46)<<iterations<<"|\n";
    std::cout<<"| Warmup Runs: "<<std::setw(45)<<warmup<<"|\n";
    std::cout<<"| Tile Size: "<<std::setw(47)<<tileSize<<"|\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"\n";
    
    // Generate test data
    const int dbCount=1000000;
    const int queryCount=16;
    const int dim=128;
    const int k=10;
    
    std::cout<<"Generating test data...\n";
    std::vector<float> db,queries;
    db.resize(dbCount*dim);
    queries.resize(queryCount*dim);
    
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f,1.0f);
    
    // Generate and normalize database
    for(int i=0;i<dbCount*dim;i++)
    {
        db[i]=dist(gen);
    }
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
    
    // Generate and normalize queries
    for(int i=0;i<queryCount*dim;i++)
    {
        queries[i]=dist(gen);
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
    
    std::cout<<"Database: "<<dbCount<<" vectors x "<<dim<<" dim\n";
    std::cout<<"Queries: "<<queryCount<<" vectors x "<<dim<<" dim\n";
    std::cout<<"Top-k: "<<k<<"\n\n";
    
    // Warmup GPU
    std::cout<<"GPU warmup...\n";
    HyperdriveANN ann;
    if(!ann.initialize(dbCount,dim,queryCount,k,tileSize))
    {
        std::cerr<<"Failed to initialize GPU ANN\n";
        return;
    }
    
    for(int w=0;w<warmup;w++)
    {
        auto gpuResults=ann.topK(db.data(),dbCount,dim,queries.data(),queryCount,k);
    }
    cudaDeviceSynchronize();
    
    // Benchmark GPU
    std::cout<<"Benchmarking GPU...\n";
    auto gpuStart=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iterations;i++)
    {
        auto gpuResults=ann.topK(db.data(),dbCount,dim,queries.data(),queryCount,k);
    }
    cudaDeviceSynchronize();
    auto gpuEnd=std::chrono::high_resolution_clock::now();
    
    auto gpuDuration=std::chrono::duration_cast<std::chrono::microseconds>(gpuEnd-gpuStart);
    double gpuMsPerQuery=gpuDuration.count()/1000.0/iterations/queryCount;
    double gpuQueriesPerSec=1000.0/gpuMsPerQuery;
    
    // Benchmark CPU
    std::cout<<"Benchmarking CPU...\n";
    auto cpuStart=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iterations;i++)
    {
        for(int q=0;q<queryCount;q++)
        {
            auto cpuResults=topKCPU(db.data(),dbCount,dim,&queries[q*dim],k);
        }
    }
    auto cpuEnd=std::chrono::high_resolution_clock::now();
    
    auto cpuDuration=std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd-cpuStart);
    double cpuMsPerQuery=cpuDuration.count()/1000.0/iterations/queryCount;
    double cpuQueriesPerSec=1000.0/cpuMsPerQuery;
    
    // Performance Analysis
    double speedup=cpuMsPerQuery/gpuMsPerQuery;
    std::string performance="EXCELLENT";
    std::string color="\033[32m"; // Green
    
    if(speedup<10)
    {
        performance="POOR";
        color="\033[31m"; // Red
    }
    else if(speedup<50)
    {
        performance="FAIR";
        color="\033[33m"; // Yellow
    }
    else if(speedup<100)
    {
        performance="GOOD";
        color="\033[36m"; // Cyan
    }
    else if(speedup<500)
    {
        performance="VERY GOOD";
        color="\033[32m"; // Green
    }
    
    // Results
    std::cout<<"+==============================================================+\n";
    std::cout<<"|                      BENCHMARK RESULTS                       |\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"| Database: "<<std::left<<std::setw(47)<<dbCount<<" vectors |\n";
    std::cout<<"| Queries: "<<std::setw(48)<<queryCount<<" vectors |\n";
    std::cout<<"| Dimensions: "<<std::setw(45)<<dim<<"|\n";
    std::cout<<"| Top-K: "<<std::setw(49)<<k<<"|\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"| GPU Performance:                                              |\n";
    std::cout<<"|   Time per query: "<<std::setw(38)<<std::fixed<<std::setprecision(3)<<gpuMsPerQuery<<" ms |\n";
    std::cout<<"|   Queries/sec: "<<std::setw(41)<<std::setprecision(0)<<gpuQueriesPerSec<<"|\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"| CPU Performance:                                              |\n";
    std::cout<<"|   Time per query: "<<std::setw(38)<<std::setprecision(3)<<cpuMsPerQuery<<" ms |\n";
    std::cout<<"|   Queries/sec: "<<std::setw(41)<<std::setprecision(0)<<cpuQueriesPerSec<<"|\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"| "<<color<<"GPU Speedup: "<<std::setw(37)<<std::setprecision(1)<<speedup<<"x "<<performance<<"\033[0m"<<std::setw(9-performance.length())<<"|\n";
    std::cout<<"+==============================================================+\n";
    
    // Verify correctness
    std::cout<<"\n=== CORRECTNESS CHECK ===\n";
    auto gpuResults=ann.topK(db.data(),dbCount,dim,queries.data(),queryCount,k);
    auto cpuResults=topKCPU(db.data(),dbCount,dim,queries.data(),k);
    
    bool correct=true;
    for(int i=0;i<k;i++)
    {
        if(gpuResults[0][i]!=cpuResults[i])
        {
            correct=false;
            break;
        }
    }
    
    std::cout<<"\n";
    std::cout<<"+==============================================================+\n";
    std::cout<<"|                     CORRECTNESS CHECK                        |\n";
    std::cout<<"+==============================================================+\n";
    
    if(correct)
    {
        std::cout<<"| "<<std::left<<"\033[32m[PASS] GPU and CPU results match perfectly!\033[0m"<<std::setw(19)<<"|\n";
    }
    else
    {
        std::cout<<"| "<<std::left<<"\033[31m[FAIL] WARNING: GPU and CPU results differ!\033[0m"<<std::setw(14)<<"|\n";
        std::cout<<"| GPU top-"<<k<<": ";
        for(int i=0;i<k;i++)
        {
            std::cout<<gpuResults[0][i];
            if(i<k-1) std::cout<<",";
        }
        std::cout<<std::setw(35-k*2)<<"|\n";
        std::cout<<"| CPU top-"<<k<<": ";
        for(int i=0;i<k;i++)
        {
            std::cout<<cpuResults[i];
            if(i<k-1) std::cout<<",";
        }
        std::cout<<std::setw(35-k*2)<<"|\n";
    }
    std::cout<<"+==============================================================+\n";
}

int main(int argc,char* argv[])
{
    if(argc<2)
    {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string mode=argv[1];
    
    if(mode=="--help")
    {
        printUsage(argv[0]);
        return 0;
    }
    else if(mode=="--go")
    {
        printBanner();
        printDeviceInfo();
        
        std::cout<<"+==============================================================+\n";
        std::cout<<"|                     QUICK DEMO MODE                          |\n";
        std::cout<<"+==============================================================+\n";
        std::cout<<"| This will test your GPU performance by:                     |\n";
        std::cout<<"| 1. Creating synthetic vector database (normalized)          |\n";
        std::cout<<"| 2. Generating test queries with same dimensions             |\n";
        std::cout<<"| 3. Finding top-k most similar vectors (cosine similarity)   |\n";
        std::cout<<"| 4. Comparing GPU vs CPU performance                        |\n";
        std::cout<<"| 5. Verifying correctness of results                        |\n";
        std::cout<<"+==============================================================+\n";
        std::cout<<"\n";
        
        // Ask for confirmation
        std::cout<<"Do you want to proceed with the test? (Y/n): ";
        std::string confirm;
        std::getline(std::cin,confirm);
        
        if(confirm!="Y"&&confirm!="y"&&confirm!="yes"&&confirm!="Yes"&&confirm!="YES"&&confirm!="")
        {
            std::cout<<"Test cancelled.\n";
            return 0;
        }
        
        std::cout<<"\n";
        std::cout<<"+==============================================================+\n";
        std::cout<<"|                    SELECT TEST SIZE                          |\n";
        std::cout<<"+==============================================================+\n";
        std::cout<<"| 1. Quick Test (100,000 vectors x 128 dim) - ~5 seconds      |\n";
        std::cout<<"| 2. Full Test (1,000,000 vectors x 128 dim) - ~30 seconds    |\n";
        std::cout<<"+==============================================================+\n";
        std::cout<<"\n";
        std::cout<<"Choose test size (1 or 2): ";
        
        int testSize;
        std::cin>>testSize;
        
        if(testSize!=1&&testSize!=2)
        {
            std::cout<<"Invalid choice. Using Quick Test.\n";
            testSize=1;
        }
        
        std::cout<<"\n";
        std::cout<<"Do you want to proceed with the test? (Y/n): ";
        std::string confirm2;
        std::cin.ignore();
        std::getline(std::cin,confirm2);
        
        if(confirm2!="Y"&&confirm2!="y"&&confirm2!="yes"&&confirm2!="Yes"&&confirm2!="YES"&&confirm2!="")
        {
            std::cout<<"Test cancelled.\n";
            return 0;
        }
        
        std::cout<<"\n";
        std::cout<<"Generating test data...\n";
        
        const int dbCount=(testSize==1)?100000:1000000;
        const int queryCount=16;
        const int dim=128;
        const int k=10;
        
        std::vector<float> db,queries;
        db.resize(dbCount*dim);
        queries.resize(queryCount*dim);
        
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f,1.0f);
        
        // Generate and normalize database
        for(int i=0;i<dbCount*dim;i++)
        {
            db[i]=dist(gen);
        }
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
        
        // Generate and normalize queries
        for(int i=0;i<queryCount*dim;i++)
        {
            queries[i]=dist(gen);
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
        
        std::cout<<"Database: "<<dbCount<<" vectors x "<<dim<<" dim\n";
        std::cout<<"Queries: "<<queryCount<<" vectors x "<<dim<<" dim\n";
        std::cout<<"Top-k: "<<k<<"\n\n";
        
        std::cout<<"Starting benchmark...\n\n";
        
        // Run benchmark
        benchmark(1,1,256);
        
        std::cout<<"\n";
        std::cout<<"+==============================================================+\n";
        std::cout<<"|                      TEST COMPLETE                          |\n";
        std::cout<<"+==============================================================+\n";
        std::cout<<"| Your system performance has been evaluated!                  |\n";
        std::cout<<"|                                                              |\n";
        std::cout<<"| Try advanced options:                                        |\n";
        std::cout<<"|   --benchmark with your own data files                       |\n";
        std::cout<<"|   --tile-size 512 for different performance                  |\n";
        std::cout<<"|   --iterations 5 for more accurate results                   |\n";
        std::cout<<"+==============================================================+\n";
        
        return 0;
    }
    else if(mode=="--generate")
    {
        if(argc!=5)
        {
            std::cout<<"Usage: "<<argv[0]<<" --generate <count> <dim> <filename>\n";
            return 1;
        }
        
        int count=std::atoi(argv[2]);
        int dim=std::atoi(argv[3]);
        std::string filename=argv[4];
        
        generateData(count,dim,filename);
    }
    else if(mode=="--benchmark")
    {
        if(argc<5)
        {
            std::cout<<"Usage: "<<argv[0]<<" --benchmark <db_file> <query_file> <k>\n";
            return 1;
        }
        
        std::string dbFile=argv[2];
        std::string queryFile=argv[3];
        int k=std::atoi(argv[4]);
        
        std::vector<float> db,queries;
        int dbCount,dbDim,queryCount,queryDim;
        
        loadData(dbFile,db,dbCount,dbDim);
        loadData(queryFile,queries,queryCount,queryDim);
        
        if(dbDim!=queryDim)
        {
            std::cerr<<"Dimension mismatch: DB="<<dbDim<<", Queries="<<queryDim<<"\n";
            return 1;
        }
        
        // Parse additional options
        int tileSize=256;
        int iterations=1;
        int warmup=1;
        
        for(int i=5;i<argc;i+=2)
        {
            if(i+1>=argc) break;
            
            std::string option=argv[i];
            if(option=="--tile-size")
            {
                tileSize=std::atoi(argv[i+1]);
            }
            else if(option=="--iterations")
            {
                iterations=std::atoi(argv[i+1]);
            }
            else if(option=="--warmup")
            {
                warmup=std::atoi(argv[i+1]);
            }
        }
        
        benchmark(iterations,warmup,tileSize);
    }
    else
    {
        std::cout<<"Unknown option: "<<mode<<"\n";
        printUsage(argv[0]);
        return 1;
    }
    
    return 0;
}
