#include"quantization.hpp"
#include<algorithm>
#include<cmath>
#include<cstring>

VectorQuantizer::VectorQuantizer():type_(QuantizationType::NONE),initialized_(false)
{
}

VectorQuantizer::~VectorQuantizer()
{
    cleanup();
}

bool VectorQuantizer::initialize(QuantizationType type,const float* vectors,int count,int dim)
{
    if(initialized_)
    {
        cleanup();
    }
    
    type_=type;
    
    if(type==QuantizationType::NONE)
    {
        initialized_=true;
        return true;
    }
    
    computeQuantizationParams(vectors,count,dim);
    initialized_=true;
    
    return true;
}

void VectorQuantizer::cleanup()
{
    scales_.clear();
    offsets_.clear();
    type_=QuantizationType::NONE;
    initialized_=false;
}

void VectorQuantizer::quantize(const float* input,int count,int dim,void* output)
{
    if(!initialized_||type_==QuantizationType::NONE)
    {
        std::memcpy(output,input,count*dim*sizeof(float));
        return;
    }
    
    switch(type_)
    {
        case QuantizationType::INT8:
            quantizeInt8(input,count,dim,static_cast<int8_t*>(output));
            break;
        case QuantizationType::INT4:
            quantizeInt4(input,count,dim,static_cast<uint8_t*>(output));
            break;
        default:
            break;
    }
}

void VectorQuantizer::dequantize(const void* input,int count,int dim,float* output)
{
    if(!initialized_||type_==QuantizationType::NONE)
    {
        std::memcpy(output,input,count*dim*sizeof(float));
        return;
    }
    
    switch(type_)
    {
        case QuantizationType::INT8:
            dequantizeInt8(static_cast<const int8_t*>(input),count,dim,output);
            break;
        case QuantizationType::INT4:
            dequantizeInt4(static_cast<const uint8_t*>(input),count,dim,output);
            break;
        default:
            break;
    }
}

size_t VectorQuantizer::getQuantizedSize(int count,int dim) const
{
    switch(type_)
    {
        case QuantizationType::INT8:
            return count*dim*sizeof(int8_t);
        case QuantizationType::INT4:
            return (count*dim*4+7)/8;
        default:
            return count*dim*sizeof(float);
    }
}

float VectorQuantizer::getCompressionRatio() const
{
    switch(type_)
    {
        case QuantizationType::INT8:
            return 4.0f;
        case QuantizationType::INT4:
            return 8.0f;
        default:
            return 1.0f;
    }
}

void VectorQuantizer::computeQuantizationParams(const float* vectors,int count,int dim)
{
    scales_.resize(dim);
    offsets_.resize(dim);
    
    for(int d=0;d<dim;d++)
    {
        float min_val=vectors[d];
        float max_val=vectors[d];
        
        for(int i=1;i<count;i++)
        {
            float val=vectors[i*dim+d];
            min_val=std::min(min_val,val);
            max_val=std::max(max_val,val);
        }
        
        offsets_[d]=min_val;
        
        if(type_==QuantizationType::INT8)
        {
            scales_[d]=(max_val-min_val)/255.0f;
        }
        else if(type_==QuantizationType::INT4)
        {
            scales_[d]=(max_val-min_val)/15.0f;
        }
    }
}

void VectorQuantizer::quantizeInt8(const float* input,int count,int dim,int8_t* output)
{
    for(int i=0;i<count;i++)
    {
        for(int d=0;d<dim;d++)
        {
            float normalized=(input[i*dim+d]-offsets_[d])/scales_[d];
            int8_t quantized=static_cast<int8_t>(std::round(std::max(0.0f,std::min(255.0f,normalized))));
            output[i*dim+d]=quantized-128;
        }
    }
}

void VectorQuantizer::quantizeInt4(const float* input,int count,int dim,uint8_t* output)
{
    int packed_size=(count*dim+1)/2;
    
    for(int i=0;i<packed_size;i++)
    {
        int idx1=i*2;
        int idx2=i*2+1;
        
        float val1=(idx1<count*dim)?(input[idx1]-offsets_[idx1%dim])/scales_[idx1%dim]:0.0f;
        float val2=(idx2<count*dim)?(input[idx2]-offsets_[idx2%dim])/scales_[idx2%dim]:0.0f;
        
        uint8_t quant1=static_cast<uint8_t>(std::round(std::max(0.0f,std::min(15.0f,val1))));
        uint8_t quant2=static_cast<uint8_t>(std::round(std::max(0.0f,std::min(15.0f,val2))));
        
        output[i]=(quant1&0xF)|((quant2&0xF)<<4);
    }
}

void VectorQuantizer::dequantizeInt8(const int8_t* input,int count,int dim,float* output)
{
    for(int i=0;i<count;i++)
    {
        for(int d=0;d<dim;d++)
        {
            int8_t quantized=input[i*dim+d];
            float dequantized=scales_[d]*(static_cast<float>(quantized+128))+offsets_[d];
            output[i*dim+d]=dequantized;
        }
    }
}

void VectorQuantizer::dequantizeInt4(const uint8_t* input,int count,int dim,float* output)
{
    int packed_size=(count*dim+1)/2;
    
    for(int i=0;i<packed_size;i++)
    {
        int idx1=i*2;
        int idx2=i*2+1;
        
        uint8_t packed=input[i];
        uint8_t quant1=packed&0xF;
        uint8_t quant2=(packed>>4)&0xF;
        
        if(idx1<count*dim)
        {
            output[idx1]=scales_[idx1%dim]*static_cast<float>(quant1)+offsets_[idx1%dim];
        }
        
        if(idx2<count*dim)
        {
            output[idx2]=scales_[idx2%dim]*static_cast<float>(quant2)+offsets_[idx2%dim];
        }
    }
}
