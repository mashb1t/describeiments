import time

import torch
from PIL import Image

from modules import args_parser
from modules.model_management import load_device, offload_device, init_model


def describe_image(image_filepaths, prompt):
    print(f"[Describe] ===== Processing start =====")
    preparation_start_time = time.perf_counter()
    processor, model = init_model()

    if prompt:
        print(f"[Describe] Raw Prompt: {prompt}")
        prompt = f"Question: {prompt} Answer:"
        print(f"[Describe] Prompt: {prompt}")

    images = [Image.open(image_filepath) for image_filepath in image_filepaths]

    params = {
        'images': images,
        'return_tensors': "pt"
    }

    if prompt:
        params['text'] = [prompt if prompt is not None else '' for _ in range(len(images))]

    inputs = processor(**params)

    moving_start_time = time.perf_counter()

    if not args_parser.args.is_quantized:
        inputs.to(load_device)
        model.to(load_device)

    moving_intermediate_time = time.perf_counter() - moving_start_time

    preparation_time = time.perf_counter() - preparation_start_time
    print(f'[Describe] Preparation time: {preparation_time:.2f} seconds')

    execution_start_time = time.perf_counter()

    generated_ids = model.generate(**inputs)
    descriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    descriptions = [i.strip() for i in descriptions]

    print(f'[Describe] Descriptions: {descriptions}')

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

    return "\n".join(descriptions)
