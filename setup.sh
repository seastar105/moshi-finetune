pip install -U pip
pip install -e .
pip install datasets transformers hf_transfer
pip install 'torch_xla[tpu]~=2.6.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip install lightning
python download_files.py