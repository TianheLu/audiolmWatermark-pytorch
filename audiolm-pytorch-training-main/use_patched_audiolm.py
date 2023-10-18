import argparse
import urllib.request
import os
import zipfile
import subprocess
import shutil

def main(branch_name):
    branch_name='personal_hacks'
    zip_file_path = f"audiolmWatermark-pytorch-{branch_name}.zip"
    # input("type anything to confirm that you have pushed the latest version of the branch to Github as well!!")

    if os.path.isfile(zip_file_path):
        replace = input(f"{branch_name} zip already exists. replacing (as well as audiolm_pytorch library...)")
        os.remove(zip_file_path)
        shutil.rmtree("audiolm_pytorch")
    urllib.request.urlretrieve(f"https://github.com/LWProgramming/audiolm-pytorch/archive/refs/heads/{branch_name}.zip", zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("audiolmWatermark-pytorch")

    # install requirements from the patched audiolmWatermark-pytorch directory
    subprocess.run(['pip', 'install', f'audiolmWatermark-pytorch/audiolmWatermark-pytorch-{branch_name}'])

    # move library itself to current directory
    subprocess.run(['mv', f'audiolmWatermark-pytorch/audiolmWatermark-pytorch-{branch_name}/audiolm_pytorch', '.'])

    # remove the rest of the audiolmWatermark-pytorch directory
    subprocess.run(['rm', '-rf', 'audiolmWatermark-pytorch'])

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Install a specific branch of the audiolmWatermark-pytorch library.")
    # parser.add_argument("branch_name", type=str, nargs="?", default="personal_hacks", help="The name of the Github branch to install.")
    # args = parser.parse_args()
    # main(args.branch_name)
    main('personal_hacks')