"""
Madison Task Queue Engine
Manages asynchronous background LLM executions, polling states, and model-priority queue scheduling.
"""

import uuid
import time
import asyncio
from typing import Dict, Any, List
from core.hardware import get_loaded_models

class TaskStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class MadisonTaskQueue:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._queue: List[str] = []
        self._running_task_id: str = None
        self._worker_task = None

    def start_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())

    async def submit_task(self, domain: str, target_id: str, runner_func, runner_kwargs: dict, preferred_model: str = "") -> str:
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        async with self._lock:
            self.tasks[task_id] = {
                "id": task_id,
                "domain": domain,              # 'chat' or 'coding'
                "target_id": target_id,        # session_id or thread_id
                "status": TaskStatus.QUEUED,
                "preferred_model": preferred_model,
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "read": False,
                "_runner": runner_func,
                "_kwargs": runner_kwargs
            }
            # Model-Grouping Queue Optimization: Place jobs sharing currently loaded model earlier in queue
            loaded = get_loaded_models()
            current_active = loaded[0] if loaded else ""
            if preferred_model and preferred_model == current_active:
                self._queue.insert(0, task_id)
            else:
                self._queue.append(task_id)
        
        self.start_worker()
        return task_id

    async def _process_queue(self):
        while True:
            task_id = None
            async with self._lock:
                if self._queue:
                    task_id = self._queue.pop(0)
                    self._running_task_id = task_id
                    if task_id in self.tasks:
                        self.tasks[task_id]["status"] = TaskStatus.RUNNING
                        self.tasks[task_id]["started_at"] = time.time()

            if not task_id:
                await asyncio.sleep(0.3)
                continue

            task = self.tasks.get(task_id)
            if task:
                try:
                    runner = task["_runner"]
                    kwargs = task["_kwargs"]
                    # Execute synchronous blocking runner in default executor thread
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda: runner(**kwargs))
                    
                    async with self._lock:
                        task["status"] = TaskStatus.COMPLETED
                        task["result"] = result
                        task["finished_at"] = time.time()
                except Exception as e:
                    async with self._lock:
                        task["status"] = TaskStatus.FAILED
                        task["error"] = str(e)
                        task["finished_at"] = time.time()
                finally:
                    async with self._lock:
                        self._running_task_id = None

            await asyncio.sleep(0.1)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        task = self.tasks.get(task_id)
        if not task:
            return None
        # Return safe copy without private callables
        return {k: v for k, v in task.items() if not k.startswith("_")}

    def mark_task_read(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id]["read"] = True

    def get_active_and_unread_states(self) -> Dict[str, Any]:
        running = []
        unread = []
        for tid, t in self.tasks.items():
            if t["status"] in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                running.append({
                    "task_id": tid,
                    "domain": t["domain"],
                    "target_id": t["target_id"],
                    "status": t["status"]
                })
            elif t["status"] in (TaskStatus.COMPLETED, TaskStatus.FAILED) and not t["read"]:
                unread.append({
                    "task_id": tid,
                    "domain": t["domain"],
                    "target_id": t["target_id"],
                    "status": t["status"]
                })
        return {"running": running, "unread": unread}

TASK_QUEUE = MadisonTaskQueue()