.PHONY: install clean fclean

install:
	uv sync

clean:
	find . -name .venv -prune -o \( -name __pycache__ -o -name .mypy_cache \
		\) -type d -prune -exec rm -rf {} +

fclean: clean
	rm -rf .venv
