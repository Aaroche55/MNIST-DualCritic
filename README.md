# MNIST-DualCritic
Advanced AI Module Project

## Development

```sh
uv sync                  # install dependencies, including the dev tools
uv run pytest            # run the test suite (offline; uses a small synthetic dataset)
uv run ruff check .      # lint
uv run ruff format .     # format
uv run data --help       # dataset CLI: download, generate transforms, inspect samples
```
