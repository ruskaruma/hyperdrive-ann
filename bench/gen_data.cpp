#include<iostream>
#include<random>
#include<vector>
#include<chrono>
#include<algorithm>
#include<cmath>
#include<cstring>

class DataGenerator
{
private:
    std::mt19937 gen_;
    std::normal_distribution<float> dist_;
    
public:
    DataGenerator():gen_(std::random_device{}()),dist_(0.0f,1.0f)
    {
    }
    
    void generateVectors(std::vector<float>& vectors,int count,int dim)
    {
        vectors.resize(count*dim);
        for(int i=0;i<count*dim;i++)
        {
            vectors[i]=dist_(gen_);
        }
        
        // Normalize vectors for cosine similarity
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
    }
    
    void saveVectors(const std::vector<float>& vectors,int count,int dim,const std::string& filename)
    {
        FILE* f=fopen(filename.c_str(),"wb");
        if(!f)
        {
            std::cerr<<"Failed to open "<<filename<<" for writing\n";
            return;
        }
        
        // Simple binary format: count, dim, data
        fwrite(&count,sizeof(int),1,f);
        fwrite(&dim,sizeof(int),1,f);
        fwrite(vectors.data(),sizeof(float),count*dim,f);
        fclose(f);
        
        std::cout<<"Saved "<<count<<" vectors of dimension "<<dim<<" to "<<filename<<"\n";
    }
    
    void loadVectors(std::vector<float>& vectors,int& count,int& dim,const std::string& filename)
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
};

float cosineSimilarity(const float* a,const float* b,int dim)
{
    float dot=0.0f;
    for(int i=0;i<dim;i++)
    {
        dot+=a[i]*b[i];
    }
    return dot; // vectors are normalized, so dot product = cosine similarity
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
    
    // Partial sort to get top k
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

std::vector<std::vector<int>> batchTopKCPU(const float* db,int dbCount,int dim,const float* queries,int queryCount,int k)
{
    std::vector<std::vector<int>> results;
    results.reserve(queryCount);
    
    for(int q=0;q<queryCount;q++)
    {
        auto topK=topKCPU(db,dbCount,dim,&queries[q*dim],k);
        results.push_back(std::move(topK));
    }
    
    return results;
}

int main(int argc,char* argv[])
{
    if(argc<2)
    {
        std::cout<<"Usage: "<<argv[0]<<" [generate|benchmark] [options]\n";
        std::cout<<"  generate <count> <dim> <filename>  - Generate random vectors\n";
        std::cout<<"  benchmark <db_file> <query_file> <k>  - Benchmark CPU vs GPU\n";
        return 1;
    }
    
    std::string mode=argv[1];
    
    if(mode=="generate")
    {
        if(argc!=5)
        {
            std::cout<<"Usage: "<<argv[0]<<" generate <count> <dim> <filename>\n";
            return 1;
        }
        
        int count=std::atoi(argv[2]);
        int dim=std::atoi(argv[3]);
        std::string filename=argv[4];
        
        DataGenerator gen;
        std::vector<float> vectors;
        gen.generateVectors(vectors,count,dim);
        gen.saveVectors(vectors,count,dim,filename);
    }
    else if(mode=="benchmark")
    {
        if(argc!=5)
        {
            std::cout<<"Usage: "<<argv[0]<<" benchmark <db_file> <query_file> <k>\n";
            return 1;
        }
        
        std::string dbFile=argv[2];
        std::string queryFile=argv[3];
        int k=std::atoi(argv[4]);
        
        DataGenerator gen;
        
        // Load database
        std::vector<float> db;
        int dbCount,dbDim;
        gen.loadVectors(db,dbCount,dbDim,dbFile);
        
        // Load queries
        std::vector<float> queries;
        int queryCount,queryDim;
        gen.loadVectors(queries,queryCount,queryDim,queryFile);
        
        if(dbDim!=queryDim)
        {
            std::cerr<<"Dimension mismatch: DB="<<dbDim<<", Queries="<<queryDim<<"\n";
            return 1;
        }
        
        // Benchmark CPU
        std::cout<<"Benchmarking CPU baseline...\n";
        auto start=std::chrono::high_resolution_clock::now();
        auto results=batchTopKCPU(db.data(),dbCount,dbDim,queries.data(),queryCount,k);
        auto end=std::chrono::high_resolution_clock::now();
        
        auto duration=std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        double msPerQuery=duration.count()/1000.0/queryCount;
        
        std::cout<<"CPU Results:\n";
        std::cout<<"  Total time: "<<duration.count()/1000.0<<" ms\n";
        std::cout<<"  Time per query: "<<msPerQuery<<" ms\n";
        std::cout<<"  Queries per second: "<<1000.0/msPerQuery<<"\n";
        
        // Show first result
        std::cout<<"First query top-"<<k<<" results: ";
        for(int i=0;i<k;i++)
        {
            std::cout<<results[0][i];
            if(i<k-1) std::cout<<",";
        }
        std::cout<<"\n";
    }
    else
    {
        std::cout<<"Unknown mode: "<<mode<<"\n";
        return 1;
    }
    
    return 0;
}
