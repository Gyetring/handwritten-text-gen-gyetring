## Folder Structure

```
sdt
├── home
│   └── mdisk/...
├── Ornamentation
│   ├── checkpoints/...
│   ├── dataloader.py
│   ├── gen.py
│   ├── hello.py
│   ├── __init__.py
│   ├── labels/...
│   ├── ornamentation_network.py
│   ├── preprocess.py
│   ├── skeletons/...
│   └── train.py
├── readme.md
├── run.py
├── SDT
│   ├── configs/...
│   ├── data -> /home/gyetring/sdt/home/mdisk/daigang/StyleWriting/data
│   ├── data_loader/...
│   ├── data_old -> /home/mdisk/daigang/StyleWriting/data
│   ├── environment.yml
│   ├── evaluate.py
│   ├── fine_trained.pth
│   ├── __init__.py
│   ├── LICENSE
│   ├── models/...
│   ├── model_zoo/...
│   ├── parse_config.py
│   ├── README.md
│   ├── Saved/...
│   ├── static/...
│   ├── test.py
│   ├── trainer
│   ├── train.py
│   ├── user_generate.py
│   └── utils/...
├── style_samples
│   └── user_0_old
└── UI.py
```

## Reproduce

### SDT

- Clone SDT repo in current working directory (`sdt`), then you should have folder `SDT`
- Download its dataset and put move it to `sdt` diroctory
- Replace link `data` in `SDT` with new link (as is shown in [Folder Structure](#folder-structure))
- Add `__init__.py` in `SDT`
- Put our `fine_trained.pth` in `SDT`
- Replace `user_generate.py` in `SDT` with ours
- Replace `configs/CHINESE_CASIA.yml` in `SDT` with ours

### Ornamentation

- You already have codes in `Ornamentation/`
- Put our `best_model.pth` in `Ornamentation/checkpoints/`
- Extract datasets `labels` and `skeletons` to `Ornamentation/` (optional)

### Enviroment

```
conda create -n sdt python=3.8 -y
conda activate sdt
# install all dependencies
conda env create -f environment.yml
```

### Run

To call the interface:

`streamlit run UI.py`

To uese the interface:

- Drag at least 15 images of clean scanned writing and generate
- Wait a few minutes...
- Query characters or sentences in Chinese
- Intermediate images will saved in `skeletons`, `std_gen`, and `ornamented`
