#!/bin/bash
nvcc -ptx kernel.cu -o kernel.ptx
nvcc cuda_driver_stream_event.cu -o cuda_driver_stream_event_example -lcuda
