# src_c/Makefile
# Phase 6: Build Automation for High-Performance C-Engine

CC = gcc
CFLAGS = -Wall -Wextra -O3
LDFLAGS = -lm
TARGET = bin/raytrace_engine
SRCS = main.c potentials.c raytracer.c batch_runner.c
OBJS = $(SRCS:.c=.o)

all: prepare $(TARGET)

prepare:
	mkdir -p bin

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c core_engine.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)