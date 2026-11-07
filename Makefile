PYTHON	= uv run python
TRAIN	= datasets/dataset_train.csv
TEST	= datasets/dataset_test.csv
WEIGHTS	= weights.json

.PHONY: all install describe describe-extra histogram scatter pair train train-sgd train-minibatch predict eval lint typecheck check \
	clean fclean re help

all: help

help:
	@echo "make install        依存パッケージをインストール (uv sync)"
	@echo "make describe       $(TRAIN) の統計を表示"
	@echo "make describe-extra ボーナスの統計も含めて表示"
	@echo "make histogram      ヒストグラムを表示"
	@echo "make scatter        散布図を表示"
	@echo "make pair           ペアプロットを表示"
	@echo "make train          $(TRAIN) で学習して $(WEIGHTS) を保存"
	@echo "make train-sgd      確率的勾配降下法で学習"
	@echo "make train-minibatch ミニバッチ勾配降下法で学習"
	@echo "make predict        $(TEST) の寮を予測して houses.csv に保存"
	@echo "make eval           $(TRAIN) の正答率を表示"
	@echo "make lint           flake8"
	@echo "make typecheck      mypy"
	@echo "make check          lint + typecheck"
	@echo "make clean          キャッシュを削除"
	@echo "make fclean         clean + .venv を削除"
	@echo "make re             fclean + install"
	@echo ""
	@echo "データセットの変更: make describe TRAIN=path/to/file.csv"

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

fclean: clean
	rm -rf .venv

re: fclean install
