#pragma once
#include <cstdio>

#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
           1000.f * ((getTickCount() - t_##name) / getTickFrequency()))
