#ifndef Log_H
#define Log_H

#include <iostream>
#include <windows.h>

#define FOREGROUND_PINK 0xd
#define FOREGROUND_LIGHT_BLUE 9


WORD changeColor(WORD attributes);

static void LOG_PREFIX(const char* format, const char* prefix, WORD attributes, va_list& args);

void LOG_ERROR(const char* format, ...);

void LOG_INFO(const char* format, ...);


void LOG_DEBUG(const char* format, ...);

#endif