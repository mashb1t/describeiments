import time

import gradio as gr
import torch

from modules import args_parser
from modules.model_management import load_device, offload_device, init_model


@torch.no_grad()
@torch.inference_mode()
def describe(image, prompt):
    print(f"[Describe] ===== Processing start =====")
    preparation_start_time = time.perf_counter()
    processor, model = init_model()

    if prompt:
        print(f"[Describe] Raw Prompt: {prompt}")
        prompt = f"Question: {prompt} Answer:"
        print(f"[Describe] Prompt: {prompt}")

    inputs = processor(images=image, text=prompt, return_tensors="pt")

    moving_start_time = time.perf_counter()

    if not args_parser.args.is_quantized:
        inputs.to(load_device)
        model.to(load_device)

    moving_intermediate_time = time.perf_counter() - moving_start_time

    preparation_time = time.perf_counter() - preparation_start_time
    print(f'[Describe] Preparation time: {preparation_time:.2f} seconds')

    execution_start_time = time.perf_counter()

    generated_ids = model.generate(**inputs)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f'[Describe] Description: {description}')

    execution_time = time.perf_counter() - execution_start_time
    print(f'[Describe] Generating time: {execution_time:.2f} seconds')

    moving_start_time = time.perf_counter()

    if not args_parser.args.is_quantized:
        model.to(offload_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    moving_time = time.perf_counter() - moving_start_time + moving_intermediate_time
    total_time = time.perf_counter() - preparation_start_time
    print(f'[Describe] Model moving time (total): {moving_time:.2f} seconds')
    print(f'[Describe] Processing time: {total_time:.2f} seconds')
    print(f'[Describe] ===== Processing done =====')

    return description


demo = gr.Interface(
    fn=describe,
    inputs=["image", gr.Textbox(label="prompt")],
    outputs=[gr.Textbox(label="description")],
)
demo.launch(
    inbrowser=args_parser.args.in_browser,
    server_name=args_parser.args.listen,
    server_port=args_parser.args.port,
    share=args_parser.args.share,
)
