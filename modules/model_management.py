import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

from modules import args_parser

processor = None
model = None

offload_device = torch.device('cpu')
load_device = torch.device(
    torch.cuda.current_device()) if torch.cuda.is_available() and not args_parser.args.always_cpu else offload_device

if args_parser.args.always_gpu and not args_parser.args.always_offload_from_vram:
    offload_device = load_device


def init_model() -> tuple[Blip2Processor, Blip2ForConditionalGeneration]:
    global processor, model
    if processor is None and model is None:
        print("[Model Management] Loading model")
        quantization_config, dtype, device_map = get_quantization_config(offload_device)
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                              torch_dtype=dtype,
                                                              device_map=device_map,
                                                              quantization_config=quantization_config)
    return processor, model


def get_quantization_config(device: torch.device) -> tuple[BitsAndBytesConfig, torch.dtype, str | torch.device]:
    quantization_config = None
    dtype = torch.float16

    if args_parser.args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        dtype = torch.float32
        device = "auto"
    elif args_parser.args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = torch.float16
        device = "auto"

    return quantization_config, dtype, device
