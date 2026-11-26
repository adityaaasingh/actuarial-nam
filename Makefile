.PHONY: test lint data train pipeline serve reproduce

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

data:
	uv run python -m anam.data.bemtpl && uv run python -m anam.data.synthetic && uv run python -m anam.data.prepare

train:
	uv run python -m anam.train_glm

pipeline:
	$(MAKE) data train

serve:
	@echo 'serve: not implemented yet (M1)'

reproduce:
	@echo 'reproduce: not implemented yet (M4)'
