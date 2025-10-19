#include"config.hpp"
#include<fstream>
#include<sstream>
#include<iostream>
#include<yaml-cpp/yaml.h>
#include<stdexcept>

std::shared_ptr<HyperdriveConfig> HyperdriveConfig::loadFromFile(const std::string& config_path)
{
    auto config=std::make_shared<HyperdriveConfig>();
    
    try
    {
        YAML::Node yaml_config=YAML::LoadFile(config_path);
        
        if(yaml_config["memory"])
        {
            auto memory=yaml_config["memory"];
            if(memory["max_memory_mb"]) config->max_memory_mb_=memory["max_memory_mb"].as<size_t>();
            if(memory["pool_size"]) config->memory_pool_size_=memory["pool_size"].as<size_t>();
            if(memory["enable_pool"]) config->enable_memory_pool_=memory["enable_pool"].as<bool>();
        }
        
        if(yaml_config["performance"])
        {
            auto perf=yaml_config["performance"];
            if(perf["tile_size"]) config->tile_size_=perf["tile_size"].as<int>();
            if(perf["max_threads_per_block"]) config->max_threads_per_block_=perf["max_threads_per_block"].as<int>();
            if(perf["enable_async"]) config->enable_async_overlap_=perf["enable_async"].as<bool>();
            if(perf["num_streams"]) config->num_streams_=perf["num_streams"].as<int>();
        }
        
        if(yaml_config["algorithm"])
        {
            auto algo=yaml_config["algorithm"];
            if(algo["quantization"])
            {
                std::string quant=algo["quantization"].as<std::string>();
                if(quant=="int8") config->quantization_=QuantizationType::INT8;
                else if(quant=="int4") config->quantization_=QuantizationType::INT4;
                else config->quantization_=QuantizationType::NONE;
            }
            if(algo["filter_type"])
            {
                std::string filter=algo["filter_type"].as<std::string>();
                if(filter=="hierarchical") config->filter_type_=FilterType::HIERARCHICAL;
                else if(filter=="approximate") config->filter_type_=FilterType::APPROXIMATE;
                else config->filter_type_=FilterType::NONE;
            }
            if(algo["early_stop_threshold"]) config->early_stop_threshold_=algo["early_stop_threshold"].as<float>();
            if(algo["max_candidates"]) config->max_candidates_=algo["max_candidates"].as<int>();
        }
        
        if(yaml_config["logging"])
        {
            auto log=yaml_config["logging"];
            if(log["enable"]) config->enable_logging_=log["enable"].as<bool>();
            if(log["level"]) config->log_level_=log["level"].as<std::string>();
            if(log["file"]) config->log_file_=log["file"].as<std::string>();
        }
        
        if(yaml_config["dataset"])
        {
            auto dataset=yaml_config["dataset"];
            if(dataset["path"]) config->dataset_path_=dataset["path"].as<std::string>();
            if(dataset["type"]) config->dataset_type_=dataset["type"].as<std::string>();
            if(dataset["normalize"]) config->normalize_vectors_=dataset["normalize"].as<bool>();
        }
        
        if(yaml_config["benchmark"])
        {
            auto bench=yaml_config["benchmark"];
            if(bench["iterations"]) config->benchmark_iterations_=bench["iterations"].as<int>();
            if(bench["warmup_runs"]) config->warmup_runs_=bench["warmup_runs"].as<int>();
            if(bench["enable_profiling"]) config->enable_profiling_=bench["enable_profiling"].as<bool>();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr<<"Error loading config: "<<e.what()<<"\n";
        return nullptr;
    }
    catch(...)
    {
        std::cerr<<"Unknown error loading config\n";
        return nullptr;
    }
    
    return config;
}

std::shared_ptr<HyperdriveConfig> HyperdriveConfig::getDefault()
{
    return std::make_shared<HyperdriveConfig>();
}

bool HyperdriveConfig::saveToFile(const std::string& config_path) const
{
    try
    {
        YAML::Node yaml_config;
        
        yaml_config["memory"]["max_memory_mb"]=max_memory_mb_;
        yaml_config["memory"]["pool_size"]=memory_pool_size_;
        yaml_config["memory"]["enable_pool"]=enable_memory_pool_;
        
        yaml_config["performance"]["tile_size"]=tile_size_;
        yaml_config["performance"]["max_threads_per_block"]=max_threads_per_block_;
        yaml_config["performance"]["enable_async"]=enable_async_overlap_;
        yaml_config["performance"]["num_streams"]=num_streams_;
        
        std::string quant_str="none";
        if(quantization_==QuantizationType::INT8) quant_str="int8";
        else if(quantization_==QuantizationType::INT4) quant_str="int4";
        yaml_config["algorithm"]["quantization"]=quant_str;
        
        std::string filter_str="none";
        if(filter_type_==FilterType::HIERARCHICAL) filter_str="hierarchical";
        else if(filter_type_==FilterType::APPROXIMATE) filter_str="approximate";
        yaml_config["algorithm"]["filter_type"]=filter_str;
        
        yaml_config["algorithm"]["early_stop_threshold"]=early_stop_threshold_;
        yaml_config["algorithm"]["max_candidates"]=max_candidates_;
        
        yaml_config["logging"]["enable"]=enable_logging_;
        yaml_config["logging"]["level"]=log_level_;
        yaml_config["logging"]["file"]=log_file_;
        
        yaml_config["dataset"]["path"]=dataset_path_;
        yaml_config["dataset"]["type"]=dataset_type_;
        yaml_config["dataset"]["normalize"]=normalize_vectors_;
        
        yaml_config["benchmark"]["iterations"]=benchmark_iterations_;
        yaml_config["benchmark"]["warmup_runs"]=warmup_runs_;
        yaml_config["benchmark"]["enable_profiling"]=enable_profiling_;
        
        std::ofstream file(config_path);
        if(!file.is_open())
        {
            std::cerr<<"Failed to open config file for writing: "<<config_path<<"\n";
            return false;
        }   
        file<<yaml_config;
        file.close();    
        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr<<"Error saving config: "<<e.what()<<"\n";
        return false;
    }
}
