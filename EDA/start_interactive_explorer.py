import subprocess

command = "start python -m bokeh serve --show interactive_explorer.py"
process = subprocess.Popen(command, shell=True, start_new_session=True).wait()