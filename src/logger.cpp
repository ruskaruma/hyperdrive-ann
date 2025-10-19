#include"logger.hpp"
#include<iostream>
#include<iomanip>
#include<sstream>

std::shared_ptr<Logger> Logger::instance_=nullptr;
std::mutex Logger::instance_mutex_;

Logger::Logger():min_level_(LogLevel::INFO),console_output_(true)
{
    format_="[%TIME%] [%LEVEL%] %MESSAGE%";
}

std::shared_ptr<Logger> Logger::getInstance()
{
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if(!instance_)
    {
        instance_=std::shared_ptr<Logger>(new Logger());
    }
    return instance_;
}

void Logger::initialize(const std::string& log_file,LogLevel level,bool console)
{
    auto logger=getInstance();
    logger->min_level_=level;
    logger->console_output_=console;
    
    if(!log_file.empty())
    {
        logger->file_stream_=std::make_unique<std::ofstream>(log_file,std::ios::app);
        if(!logger->file_stream_->is_open())
        {
            std::cerr<<"Failed to open log file: "<<log_file<<"\n";
            logger->file_stream_.reset();
        }
    }
    
    logger->info("Logger initialized");
}

void Logger::log(LogLevel level,const std::string& message)
{
    if(level<min_level_)
    {
        return;
    }
    
    writeLog(level,message);
}

void Logger::debug(const std::string& message)
{
    log(LogLevel::DEBUG,message);
}

void Logger::info(const std::string& message)
{
    log(LogLevel::INFO,message);
}

void Logger::warn(const std::string& message)
{
    log(LogLevel::WARN,message);
}

void Logger::error(const std::string& message)
{
    log(LogLevel::ERROR,message);
}

void Logger::fatal(const std::string& message)
{
    log(LogLevel::FATAL,message);
}

void Logger::cleanup()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if(file_stream_&&file_stream_->is_open())
    {
        file_stream_->close();
    }
    file_stream_.reset();
}

std::string Logger::levelToString(LogLevel level) const
{
    switch(level)
    {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARN: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::getCurrentTime() const
{
    auto now=std::chrono::system_clock::now();
    auto time_t=std::chrono::system_clock::to_time_t(now);
    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch())%1000;
    
    std::stringstream ss;
    ss<<std::put_time(std::localtime(&time_t),"%Y-%m-%d %H:%M:%S");
    ss<<'.'<<std::setfill('0')<<std::setw(3)<<ms.count();
    
    return ss.str();
}

void Logger::writeLog(LogLevel level,const std::string& message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string formatted_message=format_;
    std::string time_str=getCurrentTime();
    std::string level_str=levelToString(level);
    
    size_t pos=formatted_message.find("%TIME%");
    if(pos!=std::string::npos)
    {
        formatted_message.replace(pos,6,time_str);
    }
    
    pos=formatted_message.find("%LEVEL%");
    if(pos!=std::string::npos)
    {
        formatted_message.replace(pos,7,level_str);
    }
    
    pos=formatted_message.find("%MESSAGE%");
    if(pos!=std::string::npos)
    {
        formatted_message.replace(pos,9,message);
    }
    
    if(console_output_)
    {
        if(level>=LogLevel::ERROR)
        {
            std::cerr<<formatted_message<<"\n";
        }
        else
        {
            std::cout<<formatted_message<<"\n";
        }
    }
    
    if(file_stream_&&file_stream_->is_open())
    {
        *file_stream_<<formatted_message<<"\n";
        file_stream_->flush();
    }
}
