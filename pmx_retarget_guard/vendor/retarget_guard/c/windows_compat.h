#ifndef RETARGET_GUARD_WINDOWS_COMPAT_H
#define RETARGET_GUARD_WINDOWS_COMPAT_H
#ifdef _WIN32
#include <windows.h>
#undef near
#undef far
#undef min
#undef max
#include <time.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static int clock_gettime(int clock_id, struct timespec *value)
{
    LARGE_INTEGER counter, frequency;
    (void)clock_id;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    value->tv_sec = (time_t)(counter.QuadPart / frequency.QuadPart);
    value->tv_nsec = (long)((counter.QuadPart % frequency.QuadPart) * 1000000000LL / frequency.QuadPart);
    return 0;
}
#endif
#endif
#endif
