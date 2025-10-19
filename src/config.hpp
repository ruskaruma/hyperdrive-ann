#pragma once

#include<string>
#include<unordered_map>
#include<memory>

enum class QuantizationType
{
    NONE,
    INT8,
    INT4
};

enum class FilterType
{
    NONE,
    HIERARCHICAL,
    APPROXIMATE
};

struct HyperdriveConfig
{
    //memory configuration for managing GPU memory usage and allocation
    size_t max_memory_mb_{4096};
    size_t memory_pool_size_{1024};
    bool enable_memory_pool_{true};
    
    //performance configuration
    int tile_size_{256};
    int max_threads_per_block_{1024};
    bool enable_async_overlap_{true};
    int num_streams_{4};
    
    //algorithm configuration for quantization and filtering techniques applied to the dataset
    QuantizationType quantization_{QuantizationType::NONE};
    FilterType filter_type_{FilterType::NONE};
    float early_stop_threshold_{0.95f};
    int max_candidates_{10000};
    
    //logging configuration for tracking and debugging
    bool enable_logging_{true};
    std::string log_level_{"INFO"};
    std::string log_file_{"hyperdrive.log"};
    
    //dataset configuration for loading and preprocessing data
    std::string dataset_path_{""};
    std::string dataset_type_{"synthetic"};
    bool normalize_vectors_{true};
    
    //benchmarking configuration for performance evaluation and comparison
    int benchmark_iterations_{5};
    int warmup_runs_{3};
    bool enable_profiling_{false};
    
    static std::shared_ptr<HyperdriveConfig> loadFromFile(const std::string& config_path);
    static std::shared_ptr<HyperdriveConfig> getDefault();
    bool saveToFile(const std::string& config_path) const;
    
private:
    void parseYaml(const std::string& yaml_content);
    std::string toYaml() const;
};
