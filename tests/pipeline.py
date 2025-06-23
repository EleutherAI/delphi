import random
from collections.abc import AsyncIterable, Awaitable

import pytest

from delphi.pipeline import Pipe, Pipeline, process_wrapper


@pytest.fixture
def random_seed():
    random.seed(0)


TEST_ELEMENTS_HASHABLE = [1, "a", (1, 2, 3)]
TEST_ELEMENTS = TEST_ELEMENTS_HASHABLE + [{1: 2}, [1, 2, 3]]


async def async_generate():
    for elem in TEST_ELEMENTS_HASHABLE:
        yield elem


async def async_generate_100_times():
    for _ in range(100):
        for elem in TEST_ELEMENTS_HASHABLE:
            yield elem


def sync_generate():
    for elem in TEST_ELEMENTS_HASHABLE:
        yield elem


def sync_generate_100_times():
    for _ in range(100):
        for elem in TEST_ELEMENTS_HASHABLE:
            yield elem


TEST_LOADERS = [
    async_generate,
    async_generate_100_times,
    lambda: sync_generate,
    lambda: sync_generate_100_times,
]


def identity(x):
    return x


def test_process_wrapper_not_async():
    not_async_fn = identity
    wrapped_fn = process_wrapper(not_async_fn)
    with pytest.warns(RuntimeWarning):
        assert isinstance(wrapped_fn(1), Awaitable)


@pytest.mark.xfail
@pytest.mark.asyncio
async def test_async_fails():
    not_async_fn = identity
    wrapped_fn = process_wrapper(not_async_fn)
    assert await wrapped_fn(1) == 1


@pytest.mark.parametrize("x", TEST_ELEMENTS)
@pytest.mark.asyncio
async def test_identity(x):
    async_fn = async_identity
    wrapped_fn = process_wrapper(async_fn)
    assert await wrapped_fn(x) == x
    wrapped_fn = process_wrapper(
        async_fn, preprocess=lambda x: x, postprocess=lambda x: x
    )
    assert await wrapped_fn(x) == x


async def async_identity(x):
    return x


@pytest.mark.xfail
@pytest.mark.asyncio
async def test_process_wrapper_async():
    async_fn = async_identity
    wrapped_fn = process_wrapper(async_fn, preprocess=async_fn, postprocess=async_fn)
    assert await wrapped_fn(1) == 1


async def add_1(x):
    return x + 1


async def add_2(x):
    return x + 2


async def add_3(x):
    return x + 3


@pytest.mark.asyncio
async def test_pipe():
    pipe = Pipe(add_1, add_2, add_3)
    assert await pipe(1) == [2, 3, 4]


def unflatten(x):
    while isinstance(x, list):
        x = x[0]
    return x


@pytest.mark.parametrize("get_loader", TEST_LOADERS)
@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("convert_into_pipe", [True, False])
@pytest.mark.parametrize("max_concurrent", [1, 2, 10, 100, 1000])
@pytest.mark.asyncio
async def test_identity_pipeline(
    get_loader, repeats, convert_into_pipe, max_concurrent
):
    pipe = Pipe(async_identity) if convert_into_pipe else async_identity
    elems = []
    if isinstance(get_loader(), AsyncIterable):
        async for elem in get_loader():
            elems.append(elem)
    elif callable(get_loader()):
        for elem in get_loader()():
            elems.append(elem)
    pipeline = Pipeline(get_loader(), *([pipe] * repeats))
    assert set(
        (unflatten(elem) if convert_into_pipe else elem)
        for elem in await pipeline.run()
    ) == set(elems)

    elems_async = []

    async def add_to_list(x):
        elems_async.append(x)
        return x

    pipeline = Pipeline(get_loader(), *([add_to_list] * repeats))
    results_async = await pipeline.run(max_concurrent=max_concurrent)
    assert set(results_async) == set(elems)
    assert set(elems_async) == set(elems)
    assert len(elems_async) == len(elems) * repeats
    assert len(elems_async) == len(elems) * repeats


@pytest.mark.asyncio
async def test_pipeline_failure():
    async def raise_error(_):
        raise ValueError

    pipeline = Pipeline(async_generate(), raise_error)
    with pytest.raises(ValueError):
        await pipeline.run()
