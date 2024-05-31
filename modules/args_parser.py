import argparse

import torch

parser = argparse.ArgumentParser()

parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0")
parser.add_argument("--port", type=int, default=7860)

parser.add_argument("--disable-analytics", action='store_true', help="Disables analytics for Gradio.")
parser.add_argument("--in-browser", type=bool, default=True)

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--always-gpu", action="store_true")
vram_group.add_argument("--always-cpu", type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

quantization_group = parser.add_mutually_exclusive_group()
quantization_group.add_argument("--load-in-4bit", action="store_true")
quantization_group.add_argument("--load-in-8bit", action="store_true",
                                help="Slower compared to default or --load-in-4bit")
quantization_group.add_argument("--always-offload-from-vram", action="store_true")

parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")

parser.add_argument("--temp-path", type=str, default=None)

parser.set_defaults(
    in_browser=True,
    always_offload_from_vram=False
)

args = parser.parse_args()

if args.disable_analytics:
    import os

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to: ", args.gpu_device_id)

if args.always_cpu:
    if args.always_cpu > 0:
        torch.set_num_threads(args.always_cpu)
    print(f"Running on {torch.get_num_threads()} CPU threads")

args.is_quantized = args.load_in_4bit or args.load_in_8bit
