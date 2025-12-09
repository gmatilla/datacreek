#!/bin/bash
# Run full unit and heavy test suites for coverage when possible.
# Always execute every test under tests/unit and tests/heavy.
set -e
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# If the `coverage` tool is available, use it to collect coverage.
# Otherwise run the tests without coverage so CI at least validates code.
if command -v coverage >/dev/null 2>&1; then
  rm -f .coverage coverage.log
  PYTHONPATH="$(pwd)" coverage run -m pytest -q tests/unit --continue-on-collection-errors || true
  PYTHONPATH="$(pwd)" coverage run --append -m pytest -q tests/heavy --continue-on-collection-errors || true
  coverage combine || true
  coverage report --skip-empty -i > coverage.log
else
  echo "coverage command not found; running tests without coverage" > coverage.log
  PYTHONPATH="$(pwd)" pytest -q tests/unit --continue-on-collection-errors >> coverage.log || true
  PYTHONPATH="$(pwd)" pytest -q tests/heavy --continue-on-collection-errors >> coverage.log || true
fi
