#pragma once

#include<vector>
#include<cuda_runtime.h>

enum class QuantizationType
{
    NONE,
    INT8,
    INT4
};

class VectorQuantizer
{
private:
    QuantizationType type_;
    std::vector<float> scales_;
    std::vector<float> offsets_;
    bool initialized_;
    
public:
    VectorQuantizer();
    ~VectorQuantizer();
    
    bool initialize(QuantizationType type,const float* vectors,int count,int dim);
    void cleanup();
    
    void quantize(const float* input,int count,int dim,void* output);
    void dequantize(const void* input,int count,int dim,float* output);
    
    QuantizationType getType() const{return type_;}
    bool isInitialized() const{return initialized_;}
    
    size_t getQuantizedSize(int count,int dim) const;
    float getCompressionRatio() const;
    
private:
    void computeQuantizationParams(const float* vectors,int count,int dim);
    void quantizeInt8(const float* input,int count,int dim,int8_t* output);
    void quantizeInt4(const float* input,int count,int dim,uint8_t* output);
    void dequantizeInt8(const int8_t* input,int count,int dim,float* output);
    void dequantizeInt4(const uint8_t* input,int count,int dim,float* output);
};
