import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

projectName = "Agentic_AI"

list_of_files=[
    ".github/workflow/.gitkeep",
    f"src/{projectName}/components/__init__.py",
    f"src/{projectName}/utils/__init__.py",
    f"src/{projectName}/utils/common.py",
    f"src/{projectName}/logging/__init__.py",
    f"src/{projectName}/config/__init__.py",
    f"src/{projectName}/config/configurations.py",
    f"src/{projectName}/pipeline/__init__.py",
    f"src/{projectName}/entity/__init__.py",
    f"src/{projectName}/constants/__init__.py",
    "config/config.yaml",
    ".gitignore",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/research.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")
    
    if (not os.path.exists(filepath) or (os.path.getsize(filepath)==0)):
        with open(filepath, 'w') as f:
            pass 
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")
