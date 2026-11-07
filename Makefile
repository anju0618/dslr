PYTHON	= uv run python
TRAIN	= datasets/dataset_train.csv
TEST	= datasets/dataset_test.csv
WEIGHTS	= weights.json

.PHONY: all install describe describe-extra histogram scatter pair train train-sgd train-minibatch predict eval lint typecheck check \
	clean fclean re help

all: help

help:
	@echo "make install        install dependencies (uv sync)"
	@echo "make describe       describe $(TRAIN)"
	@echo "make describe-extra describe with the bonus statistics"
	@echo "make histogram      show the histograms"
	@echo "make scatter        show the scatter plot"
	@echo "make pair           show the pair plot"
	@echo "make train          train on $(TRAIN) and save $(WEIGHTS)"
	@echo "make train-sgd      train with stochastic gradient descent"
	@echo "make train-minibatch train with mini-batch gradient descent"
	@echo "make predict        predict houses of $(TEST) into houses.csv"
	@echo "make eval           show the accuracy on $(TRAIN)"
	@echo "make lint           flake8"
	@echo "make typecheck      mypy"
	@echo "make check          lint + typecheck"
	@echo "make clean          remove caches, $(WEIGHTS) and houses.csv"
	@echo "make fclean         clean + remove .venv"
	@echo "make re             fclean + install"
	@echo ""
	@echo "override the dataset: make describe TRAIN=path/to/file.csv"

install:
	uv sync

describe:
	$(PYTHON) src/describe.py $(TRAIN)

describe-extra:
	$(PYTHON) src/describe.py $(TRAIN) --extra

histogram:
	$(PYTHON) src/histogram.py $(TRAIN)

scatter:
	$(PYTHON) src/scatter_plot.py $(TRAIN)

pair:
	$(PYTHON) src/pair_plot.py $(TRAIN)

train:
	$(PYTHON) src/logreg_train.py $(TRAIN)

train-sgd:
	$(PYTHON) src/logreg_train.py $(TRAIN) --algo sgd

train-minibatch:
	$(PYTHON) src/logreg_train.py $(TRAIN) --algo minibatch

predict:
	$(PYTHON) src/logreg_predict.py $(TEST) $(WEIGHTS)

eval:
	$(PYTHON) src/evaluate.py $(TRAIN) $(WEIGHTS)

lint:
	uv run flake8 src

typecheck:
	uv run mypy src

check: lint typecheck

clean:
	find . -name .venv -prune -o \( -name __pycache__ -o -name .mypy_cache \
		\) -type d -prune -exec rm -rf {} +
	rm -f $(WEIGHTS) houses.csv

fclean: clean
	rm -rf .venv

re: fclean install
