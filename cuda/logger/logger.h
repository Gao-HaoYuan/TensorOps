#pragma once

#include <stdio.h>

typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
} LogLevel;

void log_internal(LogLevel level, const char* file, int line, const char* func, const char* fmt, ...);

// 编译期可关闭 DEBUG 日志
#ifdef WITIN_DEBUG
#define LOG_DEBUG(fmt, ...) log_internal(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
// cuda kernel 只支持 printf，建议调试完删除
#define LOG_KERNEL(fmt, ...) printf("[KERNEL] [%s:%d %s]: " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) ((void)0)
#define LOG_KERNEL(fmt, ...) ((void)0)
#endif

#define LOG_INFO(fmt, ...)  log_internal(LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_internal(LOG_LEVEL_WARN, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_internal(LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) log_internal(LOG_LEVEL_FATAL, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
