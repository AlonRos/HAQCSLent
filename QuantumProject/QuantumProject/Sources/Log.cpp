#include "../Include/Log.h"

WORD changeColor(WORD attributes) {
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
	WORD saved_attributes;

	/* Save current attributes */
	GetConsoleScreenBufferInfo(hConsole, &consoleInfo);
	saved_attributes = consoleInfo.wAttributes;

	SetConsoleTextAttribute(hConsole, attributes);

	return saved_attributes;
}

static void LOG_PREFIX(const char* format, const char* prefix, WORD attributes, va_list& args) {
	WORD prevAttributes = changeColor(attributes);

	size_t prefixLen = strlen(prefix);
	size_t length = prefixLen + strlen(format) + 1;

	char* newFormat = (char*)malloc(length);

	strcpy_s(newFormat, length, prefix);
	strcpy_s(newFormat + prefixLen, length, format);
	newFormat[length - 1] = 0;

	vprintf(newFormat, args);
	changeColor(prevAttributes);
}

void LOG_ERROR(const char* format, ...) {
	va_list args;
	va_start(args, format);

	LOG_PREFIX(format, "[ERROR] ", FOREGROUND_RED, args);

	va_end(args);
}

void LOG_INFO(const char* format, ...) {
	va_list args;
	va_start(args, format);

	LOG_PREFIX(format, "[INFO] ", FOREGROUND_PINK, args);

	va_end(args);
}


void LOG_DEBUG(const char* format, ...) {
	va_list args;
	va_start(args, format);

	LOG_PREFIX(format, "[DEBUG] ", FOREGROUND_LIGHT_BLUE, args);

	va_end(args);
}
