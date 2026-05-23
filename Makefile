install:
	python -m pip install --upgrade pip
	pip install black pylint
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

format:
	black src/

lint:
	black --check src/
	pylint --disable=C0114 src/

check: lint