#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>

#include "logger.h"

static LogLevel current_log_level = LOG_LEVEL_WARN;

const char* log_level_to_color(LogLevel level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "\033[36m";  // 青色
        case LOG_LEVEL_INFO:  return "\033[32m";  // 绿色
        case LOG_LEVEL_WARN:  return "\033[33m";  // 黄色
        case LOG_LEVEL_ERROR: return "\033[31m";  // 红色
        case LOG_LEVEL_FATAL: return "\033[1;31m"; // 加粗红
        default: return "\033[0m";
    }
}

const char* log_level_to_string(LogLevel level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO:  return "INFO";
        case LOG_LEVEL_WARN:  return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        case LOG_LEVEL_FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

void log_set_level(LogLevel level) {
    current_log_level = level;
}

LogLevel get_log_level_from_env(void) {
    const char *env = getenv("WITIN_LOG_LEVEL");
    if (!env) return LOG_LEVEL_WARN;

    char *endptr;
    long val = strtol(env, &endptr, 10);
    if (env == endptr || *endptr != '\0') {
        return LOG_LEVEL_WARN;
    }

    switch (val) {
        case 0: return LOG_LEVEL_DEBUG;
        case 1: return LOG_LEVEL_INFO;
        case 2: return LOG_LEVEL_WARN;
        case 3: return LOG_LEVEL_ERROR;
        case 4: return LOG_LEVEL_FATAL;
        default: return LOG_LEVEL_WARN;
    }
}

void log_internal(LogLevel level, const char* file, int line, const char* func, const char* fmt, ...) {
    log_set_level(get_log_level_from_env());
    if (level < current_log_level) return;

    const char* color = log_level_to_color(level);
    const char* reset = "\033[0m";

    // Timestamp
    time_t t = time(NULL);
    struct tm tm_info;
    localtime_r(&t, &tm_info);
    char time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_info);

    // Print header
    fprintf(stderr, "%s[%s] [%s] [%s:%d %s] ",
        color,
        time_buf,
        log_level_to_string(level),
        file, line, func);

    // Print content
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    
    // Reset color
    fprintf(stderr, "%s\n", reset);
}
