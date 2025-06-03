import pathlib
import subprocess

# Find all .py files recursively from current directory
py_files = list(pathlib.Path('.').rglob('*.py'))
py_files_str = [str(p) for p in py_files]

# Make sure we have .py files
if not py_files_str:
    print("No Python files found.")
    exit()

# Format option must come first
cmd = [
    r"C:\Users\Judit\anaconda3\envs\image-materials\Scripts\pyan3.exe",
    "--dot",
    *py_files_str,
    "--uses", "--no-defines", "--colored", "--grouped", "--annotated"
]
# "C:\Users\Judit\PhD\Coding\image-materials\run_pyan.py"

# Run and save output
with open("myuses.dot", "w", encoding="utf-8") as f:
    subprocess.run(cmd, stdout=f)




# import pathlib
# import subprocess

# # Find all .py files recursively
# py_files = list(pathlib.Path('.').rglob('*.py'))
# py_files_str = [str(p) for p in py_files]

# # Build the command
# cmd = [
#     "python", "-m", "pyan",
#     *py_files_str,
#     "--uses", "--no-defines", "--colored", "--grouped", "--annotated", "--dot"
# ]

# # Run and save output
# with open("myuses.dot", "w", encoding="utf-8") as f:
#     subprocess.run(cmd, stdout=f)
