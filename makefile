# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "rm_dataset            : remove datas from raw_dataset directory."
	@echo "segmentation_masks    : generate the labels, ie segmentation masks, from the VGG json files."
	@echo "prepared_dataset      : create train, test, & validation datasets from raw datas in raw_dataset directory."
	@echo "train                 : launch training loop for a given set of parameters from configs/params.yaml."
	@echo "install               : installs project requirements."
	@echo "install-dev           : installs development requirements."
	@echo "install-docs          : installs docs requirements."
	@echo "clean                 : cleans all unecessary files."
	@echo "clean_tracking        : cleans tracking directories : hydra & mlruns."
	@echo "build_docker          : build the "developper" mode docker image of the project."
	@echo "run_docker            : run the docker container in "developper mode" to train inside it."
	@echo "build_prod            : build the "production mode" docker image of the project."
	@echo "run_prod              : run the docker container in "production mode" to train inside it."
	@echo "mlflow                : launch mlflow ui for monitoring training experiments."
	@echo "tensorboard           : launch tensorboard ui for monitoring training experiments."
	@echo "docs                  : serve generated documentation from mkdocs."
	@echo "tests                 : run unit tests."
	@echo "mypy                  : run mypy in the src folder for type hinting checking."
	@echo "cc_report             : run radon in the src folder for code complexity report."
	@echo "raw_report            : run radon in the src folder for raw report."
	@echo "mi_report             : run radon in the src folder for maintainability index report."
	@echo "hal_report            : run radon in the src folder for hal report."
	@echo "install_precommit     : installs precommit."
	@echo "check_precommit       : check precommit."


# Datas and training
.PHONY: rm_dataset
rm_dataset:
	rm -r ./datas/raw_dataset

.PHONY: segmentation_masks
segmentation_masks:
	python src/utils/make_masks.py

.PHONY: prepared_dataset
prepared_dataset:
	python src/utils/make_dataset.py

.PHONY: train
train:
	python src/train.py

# Installation
.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

.PHONY: install-docs
install-docs:
	python -m pip install -e ".[docs]" --no-cache-dir

# Cleaning
.PHONY: clean
clean:
	bash shell/clean_pycache.sh ../template_segmentation
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E "htmlcov/*" | xargs rm -rf
	rm -f .coverage

.PHONY: clean_tracking
clean_tracking:
	find . | grep -E "mlruns/*" | xargs rm -rf
	find . | grep -E "hydra/*" | xargs rm -rf

# Docker
.PHONY: build_docker
build_docker:
	docker build \
	--build-arg USER_UID=$$(id -u) \
	--build-arg USER_GID=$$(id -g) \
	--rm \
	-f Dockerfile \
	-t segmentation_project:v1 .

.PHONY: run_docker
run_docker:
	docker run --gpus all \
	--shm-size=2g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-it \
	--rm \
	-P \
	--mount type=bind,source=$(PWD),target=/media/vorph/Datas/template_segmentation \
	-e TF_FORCE_GPU_ALLOW_GROWTH=true \
	-e TF_ENABLE_ONEDNN_OPTS=true \
	-e XLA_FLAGS='--xla_gpu_autotune_level=2' \
	segmentation_project:v1

# Docker
.PHONY: build_prod
build_prod:
	docker build \
	--build-arg USER_UID=$$(id -u) \
	--build-arg USER_GID=$$(id -g) \
	--rm \
	-f Dockerfile.prod \
	-t segmentation_project:v1_prod .

.PHONY: run_prod
run_prod:
	docker run --gpus all \
	--shm-size=2g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-it \
	--rm \
	-P \
	--mount type=bind,source=$(PWD)/configs/,target=/home/vorph/configs/ \
	--mount type=bind,source=$(PWD)/datas/,target=/home/vorph/datas/ \
	--mount type=bind,source=$(PWD)/hydra/,target=/home/vorph/hydra/ \
	--mount type=bind,source=$(PWD)/mlruns/,target=/home/vorph/mlruns/ \
	-e TF_FORCE_GPU_ALLOW_GROWTH=true \
	-e TF_ENABLE_ONEDNN_OPTS=true \
	-e XLA_FLAGS='--xla_gpu_autotune_level=2' \
	segmentation_project:v1_prod


# https://stackoverflow.com/questions/43133670/getting-docker-container-id-in-makefile-to-use-in-another-command
# I ran into the same problem and realised that makefiles take output from shell variables with the use of $$.

# Experiments monitoring
.PHONY: mlflow
mlflow:
	mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri $(PWD)/mlruns/

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir $(PWD)/mlruns/

# Documentation
.PHONY: docs
docs:
	mkdocs serve

# Tests
.PHONY: tests
tests:
	python -m pytest -v --cov

# Reporting
.PHONY: mypy
mypy:
	mypy --show-error-codes src/

.PHONY: cc_report
cc_report:
	radon cc src/

.PHONY: raw_report
raw_report:
	radon raw --summary src/

.PHONY: mi_report
mi_report:
	radon mi src/

.PHONY: hal_report
hal_report:
	radon hal src/

# Precommit
.PHONY: install_precommit
install_precommit:
	pre-commit install

.PHONY: check_precommit
check_precommit:
	pre-commit run --all
