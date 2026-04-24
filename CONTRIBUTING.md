# Contributing

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/geo-customer-segmentation.git
cd geo-customer-segmentation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
pip install black flake8 isort pytest-cov
```

## Tests

```bash
pytest tests/ -v           # full suite
pytest tests/ --cov=src    # with coverage
```

## Code Style

```bash
make format   # auto-format with black + isort
make lint     # check
```

## Structure
- New features → `src/features/`
- New models → `src/models/`
- New data sources → `src/data_engineering/`
- Tests → `tests/test_{module}.py`

## Pull Requests
1. Create a feature branch
2. Write tests for new functionality
3. Run `make test && make lint`
4. Submit PR with clear description
