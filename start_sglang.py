from fire import Fire
import subprocess


def main(cuda_devices: list[int], model_name: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"):
    assert len(cuda_devices) % 2 == 0
    device_pairs = [(cuda_devices[i], cuda_devices[i+1]) for i in range(0, len(cuda_devices), 2)]
    print("Using devices:" device_pairs)
    command = 'CUDA_VISIBLE_DEVICES={},{} python -m sglang.launch_server --model-path "{}" --port 8000 --host 0.0.0.0 --tensor-parallel-size=2 --mem-fraction-static=0.8'
    running_commands = []
    for i, (dev1, dev2) in enumerate(device_pairs):
        subprocess.Popen(command.format(dev1, dev2, model_name), shell=True)
        running_commands.append(command.format(dev1, dev2, model_name))
    try:
        
    except KeyboardInterrupt:
        for process in running_commands:
            if process.poll() is None:
                continue
            process.kill()


if __name__ == "__main__":
    Fire(main)