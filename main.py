import os
import sys
import tempfile

from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content, \
    init_temp_path

REINSTALL_ALL = False
CLEANUP_ON_LAUNCH = True


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.3.0 torchvision==0.18.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")

    if not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")


def init_args():
    from modules.args_parser import args
    return args


prepare_environment()
args = init_args()

base_path = os.path.dirname(os.path.realpath(__file__))
temp_path = os.path.join(tempfile.gettempdir(), 'describeiments')
if args.temp_path:
    path = args.temp_path

init_temp_path(temp_path)
os.environ['GRADIO_TEMP_DIR'] = temp_path

external_path = os.path.join(base_path, 'external')
os.environ['HF_HOME'] = external_path
print(f'Using external path {external_path}')

if CLEANUP_ON_LAUNCH:
    print(f'[Cleanup] Attempting to delete content of temp dir {temp_path}')
    result = delete_folder_content(temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")

from webui import *
