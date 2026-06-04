import sys
from pathlib import Path

# Ensure the project root is on sys.path so tests can import evaluator, judge, etc.
sys.path.insert(0, str(Path(__file__).parent))
