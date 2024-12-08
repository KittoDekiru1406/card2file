import os
import yaml
import requests
import tempfile
from tqdm import tqdm


def download_weights(uri, cached=None, md5=None, quiet=False):
    if uri.startswith("http"):
        return download(url=uri, quiet=quiet)
    return uri


def download(url, quiet=False):
    tmp_dir = tempfile.gettempdir()
    filename = url.split("/")[-1]
    full_path = os.path.join(tmp_dir, filename)

    if os.path.exists(full_path):
        print("Model weight {} exsits. Ignore download!".format(full_path))
        return full_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(full_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return full_path


def download_config(id):
    url = "https://vocr.vn/data/vietocr/config/{}".format(id)
    r = requests.get(url)
    config = yaml.safe_load(r.text)
    return config