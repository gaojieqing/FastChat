"""
A model worker that executes the model.
"""
import argparse
import asyncio
import os
import threading
import time
import uuid
from typing import List

import requests
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import uvicorn

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from fastchat.conversation import get_conv_template
from fastchat.model.model_adapter import (
    add_model_args,
    get_conversation_template,
)
from fastchat.utils import build_logger, pretty_print_semaphore

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

app = FastAPI()


def heart_beat_worker(obj):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        obj.send_heart_beat()


class BaseModelWorker:
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_path: str,
            model_names: List[str],
            limit_worker_concurrency: int,
            conv_template: str = None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or [model_path.split("/")[-1]]
        self.limit_worker_concurrency = limit_worker_concurrency
        if conv_template:
            self.conv = get_conv_template(conv_template)
        else:
            self.conv = get_conversation_template(model_path)
        self.conv.sep_style = int(self.conv.sep_style)
        self.tokenizer = None
        self.context_len = None
        self.call_ct = 0
        self.semaphore = None

        self.heart_beat_thread = None

    def init_heart_beat(self):
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,)
        )
        self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(self.semaphore)}. "
            f"call_ct: {self.call_ct}. "
            f"worker_id: {self.worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
                self.semaphore is None
                or self.semaphore._value is None
                or self.semaphore._waiters is None
        ):
            return 0
        else:
            return (
                    self.limit_worker_concurrency
                    - self.semaphore._value
                    + len(self.semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def count_token(self, params):
        prompt = params["prompt"]
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def get_conv_template(self):
        return {"conv": self.conv}


class ModelWorker(BaseModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_path: str,
            model_names: List[str],
            limit_worker_concurrency: int,
            no_register: bool,
            device: str,
            conv_template: str = None,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        if ("instructor" in model_path.lower()):
            from InstructorEmbedding import INSTRUCTOR
            self.model = self.tokenizer = INSTRUCTOR(
                model_path, device=device
            )
        else:
            import sentence_transformers
            self.model = self.tokenizer = sentence_transformers.SentenceTransformer(
                model_path, device=device
            )

        if not no_register:
            self.init_heart_beat()

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            texts = list(map(lambda x: x.replace("\n", " "), params["input"]))
            embeddings = self.model.encode(texts)
            ret = {
                "embedding": embeddings.tolist(),
                "token_num": len(params["input"]),
            }

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        stream_interval=args.stream_interval,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
