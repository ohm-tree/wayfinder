import json
import logging
import multiprocessing
import os
import queue
import time
from typing import Dict, List

import torch

from examples.sudoku.sudokunet import SudokuCNN
from examples.workers.worker import *


class SudokuCNNWorker(Worker):
    def __init__(self,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 gpu_set: List[int],
                 queues: Dict[str, multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            name="SudokuCNNWorker" + "_" + str(task_id),
            worker_type="SudokuCNNWorker",
            worker_idx=task_id,
            queues=queues,
            run_name=run_name,
        )
        self.config = config
        self.num = config['num_comments']
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))
        self.model = SudokuCNN().to(torch.device("cuda"))

    def validate(self, task: dict) -> bool:
        return True

    def loop(self):
        my_tasks: List[dict] = self.spin_deque_tasks(
            channel='SudokuCNNWorker',
            blocking=True,
            timeout=self.config['timeout'],
            batch_size=self.config['batch_size'],
            validate=False
        )
        self.logger.info(
            f"Received {len(my_tasks)} tasks.")

        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            return

        # We have tasks to complete.
        input_data = torch.concat([
            torch.Tensor(i['data']) for i in my_tasks
        ], dim=0).to(torch.device("cuda"))
        policy: torch.Tensor
        value: torch.Tensor
        policy, value = self.model(input_data)
        np_policy = policy.cpu().detach().numpy()
        np_value = value.cpu().detach().numpy()
        for i, p, v in zip(my_tasks, np_policy, np_value):
            i["policy"] = p
            i["value"] = v
            self.enqueue(
                response=i
            )
