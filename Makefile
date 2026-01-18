.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := rewardbench scripts analysis tests

style:
	uv run black --line-length 119 --target-version py310 $(check_dirs)
	uv run isort $(check_dirs) --profile black

quality:
	uv run flake8 --max-line-length 119 $(check_dirs)
