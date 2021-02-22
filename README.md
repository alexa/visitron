## VISITRON: Visual Semantics-aligned Interactively Trained Object-Navigator

Contains code to train VISITRON, an [Oscar](https://github.com/microsoft/Oscar)-based agent
for NDH/[CVDN](https://github.com/mmurray/cvdn) task.
Work in progress.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.


## Setup

This repo uses
- Bottom-Up Attention repo. Create a folder called `/bottom-up` and copy the code from this repo (https://github.com/peteanderson80/bottom-up-attention/tree/ec7422aa6b672ff8c8a3594805cbe269cbf29723).
- Transformers repo used for running Oscar model. Install it from here (https://github.com/huggingface/transformers/tree/067923d3267325f525f4e46f357360c191ba562e) to `/tasks/FINAL_TASK/oscar/transformers_src`.

### R4R Data Setup

Create a folder `/generate_r4r` and copy [r4r_generate_data.py](https://github.com/google-research/google-research/blob/master/r4r/r4r_generate_data.py) and [graph_utils.py](https://github.com/google-research/google-research/blob/master/r4r/graph_utils.py) to it. Refer to [this repo](https://github.com/google-research/google-research/tree/master/r4r) for other details about R4R.
