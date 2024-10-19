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
        my_tasks: List[dict] = self.spin_deque_task(
            channel=self.name,
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
        input_data = [
            policy_value_suggest_comments(
                i.task,
                i.response,
                num=self.num
            )
            for i in my_tasks
        ]
        outputs: List[RequestOutput] = self.generate(
            input_data
        )

        for i in range(len(outputs)):
            output = outputs[i].outputs[0].text
            # self.logger.info(output)
            res = parse_policy_value_output(
                output, self.logger, num=self.num)

            self.enqueue_response(
                response=res,
                task=WorkerTask(
                    head_id=my_tasks[i].head_id,
                    task_id=TaskIdentifier(
                        task_idx=my_tasks[i].task_id.task_idx,
                        task_type=PolicyValueTaskType
                    ),
                    task=my_tasks[i].task
                )
            )
