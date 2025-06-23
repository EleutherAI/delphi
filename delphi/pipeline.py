import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from functools import wraps
from typing import Any

from tqdm.asyncio import tqdm


def process_wrapper(
    function: Callable[..., Awaitable],
    preprocess: Callable | None = None,
    postprocess: Callable | None = None,
) -> Callable[..., Awaitable]:
    """
    Wraps a function with optional preprocessing and postprocessing steps.

    Args:
        function (Callable): The main function to be wrapped.
        preprocess (Callable, optional): A function to preprocess the input.
            Defaults to None.
        postprocess (Callable, optional): A function to postprocess the output.
            Defaults to None.

    Returns:
        Callable: The wrapped function.
    """

    @wraps(function)
    async def wrapped(input: Any):
        if preprocess is not None:
            input = preprocess(input)

        results = await function(input)

        if postprocess is not None:
            results = postprocess(results)

        return results

    return wrapped


class Pipe:
    """
    Represents a pipe of functions to be executed with the same input.
    """

    def __init__(self, *functions: Callable):
        """
        Initialize the Pipe with a list of functions.

        Args:
            *functions (list[Callable]): Functions to be executed in the pipe.
        """
        self.functions = functions

    async def __call__(self, input: Any) -> list[Any]:
        """
        Execute all functions in the pipe with the given input.

        Args:
            input (Any): The input to be processed by all functions.

        Returns:
            list[Any]: The results of all functions.
        """
        tasks = [function(input) for function in self.functions]

        return await asyncio.gather(*tasks)


class Pipeline:
    """
    Manages the execution of multiple pipes, handling concurrency and progress tracking.
    """

    def __init__(self, loader: AsyncIterable | Callable, *pipes: Pipe | Callable):
        """
        Initialize the Pipeline with a list of pipes.

        Args:
            loader (Callable): The loader to be executed first.
            *pipes (list[Pipe | Callable]): Pipes to be executed in the pipeline.
        """

        self.loader = loader
        self.pipes = pipes

    async def run(self, max_concurrent: int = 10) -> list[Any]:
        """
        Run the pipeline with a maximum number of concurrent tasks.

        Args:
            max_concurrent: Maximum number of concurrent tasks. Defaults to 10.

        Returns:
            list[Any]: The results of all processed items.
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = set()

        progress_bar = tqdm(desc="Processing items")
        number_of_items = 0

        async def process_and_update(item, semaphore):
            result = await self.process_item(item, semaphore)
            progress_bar.update(1)
            return result

        async for item in self.generate_items():
            number_of_items += 1
            task = asyncio.create_task(process_and_update(item, semaphore))
            tasks.add(task)

            if len(tasks) >= max_concurrent:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                results.extend(task.result() for task in done)
                tasks = pending

        if tasks:
            done, _ = await asyncio.wait(tasks)
            results.extend(task.result() for task in done)

        progress_bar.close()
        return results

    async def generate_items(self) -> AsyncIterable[Any]:
        """
        Generates items from the first pipe, which can be an async iterable or callable

        Yields:
            Any: Items generated from the first pipe.

        Raises:
            TypeError: If the first pipe is neither an async iterable nor a callable.
        """
        if isinstance(self.loader, AsyncIterable):
            async for item in self.loader:
                yield item
        elif callable(self.loader):
            for item in self.loader():
                yield item
                await asyncio.sleep(0)  # Allow other coroutines to run
        else:
            raise TypeError("The first pipe must be an async iterable or a callable")

    async def process_item(self, item: Any, semaphore: asyncio.Semaphore) -> Any:
        """
        Processes a single item through all pipes except the first one.

        Args:
            item (Any): The item to be processed.
            semaphore (asyncio.Semaphore): Semaphore for controlling concurrency.

        Returns:
            Any: The processed item.
        """
        async with semaphore:
            result = item
            for pipe in self.pipes:
                if result is not None:
                    result = await pipe(result)
                else:
                    pass
        return result
