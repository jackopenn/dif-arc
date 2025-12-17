import jax

class MetricLogger:
    def __init__(self, batch_size, prefix, buffer=True, wandb=None):
        self.batch_size = batch_size
        self.prefix = prefix
        self.buffer = buffer
        self.wandb = wandb

        self.prev_metrics = None


    def _human_format(self, num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    # https://github.com/huggingface/nanotron/blob/7bc9923285a03069ebffe994379a311aceaea546/src/nanotron/logging/base.py#L268
        if abs(num) < 1:
            return "{:.3g}".format(num)
        SIZES = ["", "K", "M", "B", "T", "P", "E"]
        num = float("{:.3g}".format(num))
        magnitude = 0
        i = 0
        while abs(num) >= 1000 and i < len(SIZES) - 1:
            magnitude += 1
            num /= 1000.0 if not divide_by_1024 else 1024.0
            i += 1
        return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[magnitude])


    def _pretty_print(self, metrics, step):
        print_string = f"step: {step}"
        for k, v in metrics.items():
            print_string += f" | {k}: {self._human_format(v)}"
        print(print_string)


    def log(self, metrics):
        if self.buffer:
            self.prev_metrics, log_metrics = metrics, self.prev_metrics 
        else:
            log_metrics = metrics
        if not log_metrics:
            return
        step = log_metrics.pop("step")
        # move to cpu - to not block 
        log_metrics = jax.tree.map(lambda x: float(x), log_metrics)
        log_metrics["samples_per_second"] = self.batch_size / log_metrics["step_time"]
        self._pretty_print(log_metrics, step)
        if self.wandb:
            log_metrics = {f"{self.prefix}/{k}": v for k, v in log_metrics.items()}
            self.wandb.log(log_metrics, step=step)