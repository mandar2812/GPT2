import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from src.gpt2.data import Dataset
from src.gpt2.training import TrainingSpec, TrainConfig, Recorder
from typing import Dict, Optional
import json
import warnings

warnings.filterwarnings(action="ignore")


class Trainer(object):
    def __init__(self, spec: TrainingSpec, config: TrainConfig):
        self.spec = spec
        self.config = config

    def save_model_config(self, path: str):
        """
        Save model configuration to a file.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.spec.model_config(), f, indent=4)
            print(f"Model config saved to {path}")

    def train(
        self,
        from_checkpoint: Optional[str] = None,
        from_pretrained: Optional[str] = None,
    ):
        if self.config.distributed:
            mp.spawn(
                self._train,
                args=(from_checkpoint, from_pretrained),
                nprocs=self.config.gpus,
            )
        else:
            self._train(0, from_checkpoint, from_pretrained)

    def _train(
        self,
        rank: int,
        from_checkpoint: Optional[str] = None,
        from_pretrained: Optional[str] = None,
    ):
        if self.config.distributed:
            torch.cuda.set_device(rank)
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:8000",
                world_size=self.config.gpus,
                rank=rank,
            )

        # Initialize training environment and prepare datasets.
        self.spec.initialize()
        train_dataset, eval_dataset = self.spec.prepare_datasets()

        # Construct a model and load its pretrained weights.
        model = self.spec.construct_model().cuda()
        if from_pretrained:
            ckpt = torch.load(from_pretrained, map_location="cuda")
            model.load_state_dict(ckpt["model"])

            # Because the weights data allocates quite a lot of GPU memories,
            # we need to free the memories explicitly.
            del ckpt
            torch.cuda.empty_cache()

        # Create an optimizer and learning rate scheduler.
        optimizer, scheduler = self.spec.create_optimizer(model.parameters())
        recorder = Recorder()

        scaler = torch.amp.GradScaler() if self.config.use_amp else None

        if self.config.distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        start_step = 0
        # Restore last training states from checkpoint.
        if from_checkpoint:
            torch.serialization.add_safe_globals([Recorder])
            ckpt = torch.load(from_checkpoint, map_location="cuda")

            start_step = int(ckpt["step"])
            print(f"Restoring training states from step {start_step}.")
            recorder = ckpt["recorder"]

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])

            train_dataset.assign(ckpt["train_dataset"])
            eval_dataset.assign(ckpt["eval_dataset"])

            if self.config.use_amp:
                scaler.load_state_dict(ckpt["amp"])

            # Because the checkpoint data allocates quite a lot of GPU
            # memories, we need to free the memories explicitly.
            del ckpt
            torch.cuda.empty_cache()

        if rank == 0:
            # Create tqdm iterator in master process to show the progress of
            # training.
            training_iters = tqdm.tqdm(
                range(start_step + 1, self.config.total_steps),
                total=self.config.total_steps - start_step - 1,
                desc=self.config.description,
                dynamic_ncols=True,
            )
            # training_iters.update(start_step + 1)
            # Save the model config to a json file
            self.save_model_config(self.config.save_model_path.replace(".pth", ".json"))
        else:
            # In other processes, use simple iterator rather than tqdm one.
            training_iters = range(start_step + 1, self.config.total_steps)

        for step in training_iters:
            # Clear CUDA cache which is used for training.
            torch.cuda.empty_cache()

            recorder.record(
                self._train_step(rank, train_dataset, model, optimizer, scheduler, scaler),
                scope="train",
            )

            # Clear CUDA cache which is used for evaluation.
            torch.cuda.empty_cache()

            if (step + 1) % self.config.eval_steps == 0:
                recorder.record(
                    self._eval_step(rank, eval_dataset, model), scope="eval"
                )
                recorder.stamp(step)

                if rank == 0:
                    training_iters.set_postfix_str(
                        recorder.format(self.config.log_format)
                    )

            # Save training states to checkpoint file.
            if rank == 0 and (step + 1) % self.config.save_steps == 0:
                ckpt = {
                    "step": step,
                    "recorder": recorder,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_dataset": train_dataset.where(),
                    "eval_dataset": eval_dataset.where(),
                }

                if self.config.use_amp:
                    ckpt["amp"] = scaler.state_dict()

                torch.save(ckpt, self.config.save_checkpoint_path)

                # Because the checkpoint data allocates quite a lot of GPU
                # memories, we need to free the memories explicitly.
                del ckpt
                torch.cuda.empty_cache()

            if (
                rank == 0 and
                self.config.save_version_steps > 0 and
                (step + 1) % self.config.save_version_steps == 0
            ):
                versioned_model_path = self.config.save_model_path.replace(
                    ".pth", f"-{step + 1}.pth"
                )
                torch.save(
                    {"model": model.state_dict(), "metrics": recorder.metrics},
                    versioned_model_path,
                )
                torch.cuda.empty_cache()

        if self.config.distributed:
            model = model.module

        if rank == 0:
            torch.save(
                {"model": model.cpu().state_dict(), "metrics": recorder.metrics},
                self.config.save_model_path,
            )

    def _train_step(
        self,
        rank: int,
        dataset: Dataset,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        scaler: Optional[torch.amp.GradScaler]
    ) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()
        data = self._fetch_from(dataset, rank, self.config.batch_train)
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                metrics = self.spec.train_objective(data, model)
                loss = metrics["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            data = self._fetch_from(dataset, rank, self.config.batch_train)
            metrics = self.spec.train_objective(data, model)
            loss = metrics["loss"]
            loss.backward()
            optimizer.step()

        scheduler.step()

        return {k: self._to_value(v) for k, v in metrics.items()}

    @torch.no_grad()
    def _eval_step(
        self, rank: int, dataset: Dataset, model: nn.Module
    ) -> Dict[str, float]:
        model.eval()

        data = self._fetch_from(dataset, rank, self.config.batch_eval)
        metrics = self.spec.eval_objective(data, model)

        return {k: self._to_value(v) for k, v in metrics.items()}

    def _fetch_from(
        self, dataset: Dataset, rank: int, batch: int
    ) -> Dict[str, torch.Tensor]:
        if self.config.distributed:
            # In distributed training environment, each process must ignore
            # sub-batches of other processes and fetch corresponding one only.
            batch = batch // self.config.gpus

            dataset.skip(rank * batch)
            data = dataset.fetch(batch)
            dataset.skip((self.config.gpus - rank - 1) * batch)
        else:
            data = dataset.fetch(self.config.batch_train)

        return {k: v.cuda() for k, v in data.items()}

    def _to_value(self, tensor: torch.Tensor) -> float:
        if self.config.distributed:
            tensor = tensor.clone()
            dist.all_reduce(tensor, op=dist.reduce_op.SUM)
            return (tensor / self.config.gpus).item()
        else:
            return tensor.item()
