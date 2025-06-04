"""
Adapted from https://github.com/linkedin/Liger-Kernel/blob/4a2da9928b1a867079276467d03f4364c33ea673/examples/huggingface/callback.py
"""
import time
from dataclasses import dataclass

import torch
from transformers import (
    TrainerControl,
    TrainerState,
    TrainingArguments
)
from transformers.integrations.integration_utils import MLflowCallback
from liger_kernel.utils import infer_device

# https://simple.wikipedia.org/wiki/Byte
# For memory, we use binary system
M_BIN_UNIT = 2**20
# For metrics (tflops), we use decimal system
T_DEC_UNIT = 10**12


def round_to_n_decimal(x, n):
    return round(x, n)


@dataclass
class Precision:
    """
    Precision is a dataclass to store the number of decimal points for each metric.
    """

    n_decimal_time: int
    n_decimal_memory: int
    n_decimal_TPS: int


@dataclass
class State:
    """
    State is a dataclass to store the internal state of the efficiency callback.
    """

    n_warmup_steps: int = 0
    total_peak_memory_allocated: float = float("-inf")
    total_peak_memory_reserved: float = float("-inf")

    step_start_time: float = 0.0
    elapsed_time: float = 0.0

    elapsed_step: int = 0

    step_start_tokens_seen: int = 0
    elapsed_tokens_seen: int = 0

    global_start_step: int = 0


@dataclass
class Time:
    """
    Time is a dataclass to store the time-related metrics.
    """

    step: int = 0
    step_time_sec: float = 0.0
    avg_step_time_sec: float = 0.0
    time_to_completion_sec: float = 0.0
    estimated_total_time_sec: float = 0.0


@dataclass
class Memory:
    """
    Memory is a dataclass to store the memory-related metrics.
    """

    step_peak_memory_allocated_MB: float = 0.0
    step_peak_memory_reserved_MB: float = 0.0
    total_peak_memory_allocated_MB: float = 0.0
    total_peak_memory_reserved_MB: float = 0.0


@dataclass
class TPS:
    """
    TPS is a dataclass to store the tokens per second metrics.
    """

    step_tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0


class EfficiencyMLflowCallback(MLflowCallback):
    """
    A callback that extends EfficiencyCallback and MLflowCallback to log efficiency metrics during training.

    This callback logs various efficiency metrics such as step time, average step time, memory usage, and tokens per second to MLflow. 
    It ensures that only the main process (world process zero) performs the logging to avoid multiple MLflow tracking logs.

    Args:
        n_warmup_steps (int, optional): Number of warmup steps before logging efficiency metrics. Defaults to 2.
        n_decimal_time (int, optional): Number of decimal places for time metrics. Defaults to 2.
        n_decimal_memory (int, optional): Number of decimal places for memory metrics. Defaults to 2.
        n_decimal_TPS (int, optional): Number of decimal places for tokens per second metrics. Defaults to 2.

    Attributes:
        state (State): State object to track various metrics and steps.
        precision (Precision): Precision object to define decimal places for metrics.
        time (Time): Time object to store time-related metrics.
        memory (Memory): Memory object to store memory-related metrics.
        tps (TPS): TPS object to store tokens per second metrics.
        device (str): Device type inferred from the environment.
        logged_params (set): Set to keep track of logged parameters to avoid re-logging.

    Methods:
        on_init_end(args, state, control, **kwargs):
            Checks training arguments to ensure proper logging setup.

        on_train_begin(args, state, control, model=None, **kwargs):
            Initializes the global start step and calls the parent method.

        on_log(args, state, control, logs, model=None, **kwargs):
            Logs efficiency metrics to MLflow if the current process is the main process.

        on_step_begin(args, state, control, **kwargs):
            Resets peak memory stats and starts the timer for the step.

        on_step_end(args, state, control, **kwargs):
            Calculates efficiency metrics for the step and logs them to MLflow if the current process is the main process.
    """
    def __init__(self, n_warmup_steps=2, n_decimal_time=2, n_decimal_memory=2, n_decimal_TPS=2, dataset_name="", model_name=""):
        super().__init__()
        self.state = State(n_warmup_steps)
        self.precision = Precision(n_decimal_time, n_decimal_memory, n_decimal_TPS)
        self.time = Time()
        self.memory = Memory()
        self.tps = TPS()
        self.device = infer_device()
        self.logged_params = set()
        self.mlflow_tags = {"model": model_name, "dataset": dataset_name}

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not args.include_num_input_tokens_seen:
            raise Exception('Please pass training argument "--include_num_input_tokens_seen" to track tokens per second')
        if args.logging_steps != 1:
            raise Exception("Please set logging_steps=1 to track the efficiency metrics accurately")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.is_world_process_zero:
            super().on_train_begin(args, state, control, model)
            self.state.global_start_step = state.global_step

            # set up mlflow tags
            self._ml_flow.set_tags(self.mlflow_tags)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict[str, float], model=None, **kwargs):
        if state.is_world_process_zero:
            if state.global_step < (self.state.global_start_step + self.state.n_warmup_steps):
                return
            else:
                logs.update(self.time.__dict__)
                logs.update(self.memory.__dict__)
                logs.update(self.tps.__dict__)
                super().on_log(args, state, control, logs, model, **kwargs)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            getattr(torch, self.device).reset_peak_memory_stats()
            self.state.step_start_time = time.perf_counter()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step < (self.state.global_start_step + self.state.n_warmup_steps):
            self.state.step_start_tokens_seen = state.num_input_tokens_seen
            return

        current_time = time.perf_counter()
        step_time = current_time - self.state.step_start_time
        self.state.elapsed_time += step_time

        global_step = state.global_step
        self.state.elapsed_step += 1
        avg_step_time = self.state.elapsed_time / self.state.elapsed_step

        self.time.step = global_step
        self.time.step_time_sec = round_to_n_decimal(step_time, self.precision.n_decimal_time)
        self.time.avg_step_time_sec = round_to_n_decimal(avg_step_time, self.precision.n_decimal_time)
        self.time.time_to_completion_sec = round_to_n_decimal(avg_step_time * (state.max_steps - global_step), self.precision.n_decimal_time)
        self.time.estimated_total_time_sec = round_to_n_decimal(avg_step_time * state.max_steps, self.precision.n_decimal_time)

        step_peak_memory_allocated = getattr(torch, self.device).memory.max_memory_allocated()
        step_peak_memory_reserved = getattr(torch, self.device).memory.max_memory_reserved()

        self.memory.step_peak_memory_allocated_MB = round_to_n_decimal(step_peak_memory_allocated / M_BIN_UNIT, self.precision.n_decimal_memory)
        self.state.total_peak_memory_allocated = max(self.state.total_peak_memory_allocated, step_peak_memory_allocated)
        self.memory.total_peak_memory_allocated_MB = round_to_n_decimal(self.state.total_peak_memory_allocated / M_BIN_UNIT, self.precision.n_decimal_memory)

        self.memory.step_peak_memory_reserved_MB = round_to_n_decimal(step_peak_memory_reserved / M_BIN_UNIT, self.precision.n_decimal_memory)
        self.state.total_peak_memory_reserved = max(self.state.total_peak_memory_reserved, step_peak_memory_reserved)
        self.memory.total_peak_memory_reserved_MB = round_to_n_decimal(self.state.total_peak_memory_reserved / M_BIN_UNIT, self.precision.n_decimal_memory)

        step_tokens_seen = state.num_input_tokens_seen - self.state.step_start_tokens_seen
        self.state.elapsed_tokens_seen += step_tokens_seen

        self.tps.step_tokens_per_second = round_to_n_decimal(step_tokens_seen / step_time, self.precision.n_decimal_TPS)
        self.tps.avg_tokens_per_second = round_to_n_decimal(self.state.elapsed_tokens_seen / self.state.elapsed_time, self.precision.n_decimal_TPS)

        self.state.step_start_tokens_seen = state.num_input_tokens_seen

        # Log efficiency metrics to MLflow
        efficiency_metrics = {
            "step_time_sec": self.time.step_time_sec,
            "avg_step_time_sec": self.time.avg_step_time_sec,
            "time_to_completion_sec": self.time.time_to_completion_sec,
            "estimated_total_time_sec": self.time.estimated_total_time_sec,
            "step_peak_memory_allocated_MB": self.memory.step_peak_memory_allocated_MB,
            "total_peak_memory_allocated_MB": self.memory.total_peak_memory_allocated_MB,
            "step_peak_memory_reserved_MB": self.memory.step_peak_memory_reserved_MB,
            "total_peak_memory_reserved_MB": self.memory.total_peak_memory_reserved_MB,
            "step_tokens_per_second": self.tps.step_tokens_per_second,
            "avg_tokens_per_second": self.tps.avg_tokens_per_second,
        }

        if state.is_world_process_zero:
            for key, value in efficiency_metrics.items():
                if key not in self.logged_params:
                    self._ml_flow.log_param(key, value)
                    self.logged_params.add(key)
                else:
                    self._ml_flow.log_metric(key, value, step=state.global_step)