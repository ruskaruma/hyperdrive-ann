#pragma once

#include<string>
#include<fstream>
#include<memory>
#include<mutex>
#include<chrono>

enum class LogLevel
{
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

class Logger
{
private:
    std::unique_ptr<std::ofstream> file_stream_;
    std::mutex mutex_;
    LogLevel min_level_;
    std::string format_;
    bool console_output_;
    
    static std::shared_ptr<Logger> instance_;
    static std::mutex instance_mutex_;
    
    Logger();
    
public:
    static std::shared_ptr<Logger> getInstance();
    static void initialize(const std::string& log_file="hyperdrive.log",LogLevel level=LogLevel::INFO,bool console=true);
    
    void log(LogLevel level,const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);
    void fatal(const std::string& message);
    
    void setLevel(LogLevel level){min_level_=level;}
    LogLevel getLevel() const{return min_level_;}
    
    void cleanup();
    
private:
    std::string levelToString(LogLevel level) const;
    std::string getCurrentTime() const;
    void writeLog(LogLevel level,const std::string& message);
};

#define LOG_DEBUG(msg) Logger::getInstance()->debug(msg)
#define LOG_INFO(msg) Logger::getInstance()->info(msg)
#define LOG_WARN(msg) Logger::getInstance()->warn(msg)
#define LOG_ERROR(msg) Logger::getInstance()->error(msg)
#define LOG_FATAL(msg) Logger::getInstance()->fatal(msg)
