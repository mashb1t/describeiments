import gradio as gr

from modules import args_parser
from modules.inference import describe_image

demo = gr.Interface(
    fn=describe_image,
    inputs=[gr.File(label="Images", file_count="multiple", file_types=['image']), gr.Textbox(label="prompt")],
    outputs=[gr.Textbox(label="description")],
)
demo.launch(
    inbrowser=args_parser.args.in_browser,
    server_name=args_parser.args.listen,
    server_port=args_parser.args.port,
    share=args_parser.args.share,
)
