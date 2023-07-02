#!/usr/bin/env python

from __future__ import annotations

import gradio as gr
import numpy as np

from model import Model

DESCRIPTION = '# [Projected GAN](https://github.com/autonomousvision/projected_gan)'


def get_sample_image_url(name: str) -> str:
    sample_image_dir = 'https://huggingface.co/spaces/hysts/projected_gan/resolve/main/samples'
    return f'{sample_image_dir}/{name}.jpg'


def get_sample_image_markdown(name: str) -> str:
    url = get_sample_image_url(name)
    return f'''
    - size: 256x256
    - seed: 0-99
    - truncation: 0.7
    ![sample images]({url})'''


model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('App'):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(label='Model',
                                             choices=model.MODEL_NAMES,
                                             value=model.MODEL_NAMES[8])
                    seed = gr.Slider(label='Seed',
                                     minimum=0,
                                     maximum=np.iinfo(np.uint32).max,
                                     step=1,
                                     value=0)
                    psi = gr.Slider(label='Truncation psi',
                                    minimum=0,
                                    maximum=2,
                                    step=0.05,
                                    value=0.7)
                    run_button = gr.Button('Run')
                with gr.Column():
                    result = gr.Image(label='Result', elem_id='result')

        with gr.TabItem('Sample Images'):
            with gr.Row():
                model_name2 = gr.Dropdown(label='Model',
                                          choices=model.MODEL_NAMES,
                                          value=model.MODEL_NAMES[0])
            with gr.Row():
                text = get_sample_image_markdown(model_name2.value)
                sample_images = gr.Markdown(text)

    model_name.change(fn=model.set_model, inputs=model_name)
    run_button.click(fn=model.set_model_and_generate_image,
                     inputs=[
                         model_name,
                         seed,
                         psi,
                     ],
                     outputs=result)
    model_name2.change(fn=get_sample_image_markdown,
                       inputs=model_name2,
                       outputs=sample_images)

demo.queue(max_size=10).launch()
