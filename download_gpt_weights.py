import os
import urllib.request
import json
import numpy as np

import tensorflow as tf
from tqdm import tqdm


def download_model_weights(model_params, models_dir):
    sizes_bag = ("124M", "355M")
    if model_params not in sizes_bag:
        raise ValueError(f"Illegal Model size.")
    
    model_dir = os.path.join(models_dir, model_params)
    os.makedirs(model_dir, exist_ok=True)

    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "model.ckpt.data-00000-of-00001", "model.ckpt.meta", "model.ckpt.index",
        "checkpoint", "encoder.json", "hparams.json","vocab.bpe"
    ]

    for filename in filenames:
        file_url = os.path.join(base_url, model_params, filename)
        file_path = os.path.join(model_dir, filename)
        download_files(file_url, file_path)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_files(url, des):

    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get("Content-Length", 0))

            if os.path.exists(des):
                file_size_local = os.path.getsize(des)
                if file_size == file_size_local:
                    print(f"File already exists: {des}")
                    return

            # 采用跟下载文件大小相同的块大小1KB
            block_size = 1024  

            progress_bar_name = os.path.basename(url)  
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_name) as progress_bar:
                with open(des, "wb") as file:
                    while True:
                        ck = response.read(block_size)
                        if not ck:
                            break
                        file.write(ck)
                        progress_bar.update(len(ck))  # Update progress bar
    except urllib.error.HTTPError:
        s = (
            f"Internet connection fliled:({url})"
            "\nTry VPN first please. Otherwise, please try again later."
            )
        print(s)


# 这个加载函数引用自书本 ：
def load_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # 'model/' 前缀移除
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
