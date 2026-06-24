"""
Created on Sat Jun 21 20:53:30 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
import gzip
import bz2
import lzma
import zipfile
import csv
import yaml
from bs4 import BeautifulSoup

###############################################################################
# Txt Reader
###############################################################################

def read_text_file(file_path, encoding="utf-8"):
    with open(file_path, "rt", encoding=encoding) as f:
        return f.read()

###############################################################################
# Json Reader
###############################################################################
def read_json_file(file_path, encoding="utf-8"):
    with open(file_path, "rt", encoding=encoding) as f:
        return json.load(f)

###############################################################################
# Jsonl Reader
###############################################################################
def read_jsonl_file(file_path, encoding="utf-8"):
    with open(file_path, "rt", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
                
###############################################################################
#  CSV Reader
###############################################################################

def read_csv_file(file_path, encoding="utf-8"):
    with open(file_path, "rt", encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

###############################################################################
# 茶 YAML Reader
###############################################################################

def read_yaml_file(file_path, encoding="utf-8"):
    with open(file_path, "rt", encoding=encoding) as f:
        return yaml.safe_load(f)

###############################################################################
#  HTML Reader
###############################################################################

def read_html_file(file_path, encoding="utf-8"):
    with open(file_path, "rt", encoding=encoding) as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()

###############################################################################
# 易 File Reader 
###############################################################################

def read_file(file_path, encoding="utf-8"):
    """
    Dispatches reading of .txt, .json, .jsonl files with optional compression.
    Returns:
        str, dict, list depending on file type.
    """
    path = str(file_path).lower()

    if path.endswith(".txt"):
        return read_text_file(file_path, encoding=encoding)
    elif path.endswith(".json"):
        return read_json_file(file_path, encoding=encoding)
    elif path.endswith(".jsonl"):
        return list(read_jsonl_file(file_path, encoding=encoding))
    elif path.endswith(".csv") or any(path.endswith(ext) for ext in [".csv.gz", ".csv.bz2"]):
        return read_csv_file(file_path, encoding=encoding)
    
    elif path.endswith(".yaml") or path.endswith(".yml") or any(path.endswith(ext) for ext in [".yaml.gz", ".yaml.bz2"]):
        return read_yaml_file(file_path, encoding=encoding)
    
    elif path.endswith(".html") or any(path.endswith(ext) for ext in [".html.gz", ".html.bz2"]):
        return read_html_file(file_path, encoding=encoding)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
        
###############################################################################
# 易 Open Compressed File 
###############################################################################

def open_compressed_file(file_path, mode="rt", encoding="utf-8"):
    """
    Opens a compressed file with proper handler.
    Supports: .gz, .bz2, .xz, .lzma, .zip
    """
    if str(file_path).endswith(".gz"):
        return gzip.open(file_path, mode, encoding=encoding)
    elif str(file_path).endswith(".bz2"):
        return bz2.open(file_path, mode, encoding=encoding)
    elif str(file_path).endswith((".xz", ".lzma")):
        return lzma.open(file_path, mode, encoding=encoding)
    elif str(file_path).endswith(".zip"):
        zf = zipfile.ZipFile(file_path)
        name = zf.namelist()[0]
        return zf.open(name, mode.replace("t", ""))  # zip only supports binary
    else:
        raise ValueError(f"Unsupported compression format: {file_path}")