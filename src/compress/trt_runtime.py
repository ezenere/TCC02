"""Minimal TensorRT engine runner backed by torch CUDA tensors (no cuda-python).

    runner = TRTRunner("runs/eixo1_resnet50_s0/trt/model_int8.engine")
    logits = runner(images_cuda_nchw_float32)      # torch tensor on cuda
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTRunner:
    def __init__(self, engine_path: str | Path):
        with open(engine_path, "rb") as fh, trt.Runtime(LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.max_batch = self.engine.get_tensor_profile_shape(self.in_name, 0)[2][0]
        self._out = None
        # Dedicated stream: on the default stream TensorRT inserts extra
        # cudaStreamSynchronize calls in enqueueV3, which inflates latency.
        self.stream = torch.cuda.Stream()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW, contiguous, float32 on the GPU — TensorRT reads raw device pointers.
        x = x.to("cuda", torch.float32).contiguous()
        if x.shape[0] > self.max_batch:
            raise ValueError(f"batch {x.shape[0]} > engine max batch {self.max_batch}")
        self.ctx.set_input_shape(self.in_name, tuple(x.shape))
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        if self._out is None or tuple(self._out.shape) != out_shape:
            self._out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
        self.ctx.set_tensor_address(self.in_name, x.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self._out.data_ptr())
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)                 # the input was produced on the caller's stream
        ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        caller.wait_stream(self.stream)                 # the caller may read the output safely
        return self._out
