PYTHON	= uv run python
TRAIN	= datasets/dataset_train.csv
TEST	= datasets/dataset_test.csv

.PHONY: all install describe describe-extra histogram scatter pair lint typecheck check \
	clean fclean re help

all: help

help:
	@echo "make install        install dependencies (uv sync)"
	@echo "make describe       describe $(TRAIN)"
	@echo "make describe-extra describe with the bonus statistics"
	@echo "make histogram      show the histograms"
	@echo "make scatter        show the scatter plot"
	@echo "make pair           show the pair plot"
	@echo "make lint           flake8"
	@echo "make typecheck      mypy"
	@echo "make check          lint + typecheck"
	@echo "make clean          remove caches"
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

lint:
	uv run flake8 src

typecheck:
	uv run mypy src

check: lint typecheck

clean:
	find . -name .venv -prune -o \( -name __pycache__ -o -name .mypy_cache \
		\) -type d -prune -exec rm -rf {} +

fclean: clean
	rm -rf .venv

re: fclean install
