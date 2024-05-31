import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from modules import args_parser

processor = None
model = None

offload_device = torch.device('cpu')
load_device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() and not args_parser.args.always_cpu else offload_device

if args_parser.args.always_gpu and not args_parser.args.always_offload_from_vram:
    offload_device = load_device


def init_model() -> tuple[Blip2Processor, Blip2ForConditionalGeneration]:
    global processor, model
    if processor is None and model is None:
        print("[Model Management] Loading model")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map=offload_device)

    return processor, model
