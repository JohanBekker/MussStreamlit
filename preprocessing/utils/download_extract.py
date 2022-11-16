from tqdm.auto import tqdm

import json
import shutil
from pathlib import Path
import requests
import tarfile
import gzip
import os

from preprocessing.utils.helpers import log_action


def untar(compressed_path, output_dir):
    filename = os.path.basename(compressed_path)
    with tarfile.open(compressed_path) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, output_dir)
    dirs1 = os.listdir(output_dir)
    dirs = os.listdir(output_dir / dirs1[0])
    # Move model files to the output_dir from the directory created by untar
    for file in dirs:
        shutil.move(output_dir / dirs1[0] / file, output_dir)
    # Remove empty untar directory, compressed model and temp directory
    os.system('rmdir "%s"' % Path(output_dir / dirs1[0]))


def ungzip(compressed_path, output_dir):
    filename = os.path.basename(compressed_path)
    assert filename.endswith('.gz')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename[:-3])
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download(url, compressed_filepath):
    if not compressed_filepath.exists():
        with log_action(f"Downloading {url.split('/')[-1]}"):
            with requests.get(url, stream=True) as request:
                try:
                    total_length = int(request.headers.get("Content-Length"))
                    with tqdm.wrapattr(request.raw, "read", total=total_length, desc="") as raw:
                        with open(compressed_filepath, 'wb') as output:
                            shutil.copyfileobj(raw, output)
                except:
                    if '.json' in str(compressed_filepath):
                        with open(compressed_filepath, 'w') as output:
                            json.dump(request.json(), output)
                    else:
                        with open(compressed_filepath, 'wb') as output:
                            output.write(request.content)
                            #shutil.copyfileobj(request.raw, output)


def download_and_extract(url, target_path):
    tmp_dir = Path('temp/')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    compressed_filepath = tmp_dir / url.split('/')[-1]

    download(url, compressed_filepath)

    extensions_to_functions = {
        '.tar.gz': untar,
        # '.tar.bz2': untar,
        # '.tgz': untar,
        # '.zip': unzip,
        '.gz': ungzip,
        # '.bz2': unbz2,
    }

    def get_extension(filename, extensions):
        possible_extensions = [ext for ext in extensions if filename.endswith(ext)]
        if len(possible_extensions) == 0:
            raise Exception(f'File {filename} has an unknown extension')
        # Take the longest (.tar.gz should take precedence over .gz)
        return max(possible_extensions, key=lambda ext: len(ext))

    extension = get_extension(os.path.basename(compressed_filepath), list(extensions_to_functions))
    extract_function = extensions_to_functions[extension]

    if not target_path.exists():
        with log_action(f"Extracting {url.split('/')[-1]}"):
            extract_function(compressed_filepath, target_path)

    compressed_filepath.unlink()
    os.system('rmdir "%s"' % Path(tmp_dir))
