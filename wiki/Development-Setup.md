# Development Setup Guide

Complete guide to setting up a development environment for the CNN Digit Recognizer.

## Prerequisites

- **Python 3.11** (recommended) or 3.7+
- **Git** for version control
- **Virtual environment** (venv or conda)
- **Text editor or IDE** (VS Code recommended)
- **4 GB RAM minimum** for development
- **1 GB disk space** for repository

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/w1setown/CNN-Digit-Recognizer
cd CNN-Digit-Recognizer
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Development Tools

```bash
pip install pytest pytest-cov black flake8 mypy jupyter
```

**What each tool does:**
- **pytest** - Unit testing
- **pytest-cov** - Coverage reporting
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **jupyter** - Interactive notebooks

### 5. Verify Installation

```bash
python diagnose.py
python verify_setup.py
```

Should see green checkmarks for all items.

---

## IDE Setup

### Visual Studio Code (Recommended)

#### Extensions to Install

1. **Python**
   - Publisher: Microsoft
   - ID: ms-python.python

2. **Pylance**
   - Publisher: Microsoft
   - ID: ms-python.vscode-pylance

3. **Jupyter**
   - Publisher: Microsoft
   - ID: ms-toolsai.jupyter

4. **Black Formatter**
   - Publisher: Microsoft
   - ID: ms-python.black-formatter

5. **Flake8**
   - Publisher: Microsoft
   - ID: ms-python.flake8

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true
    }
}
```

#### VS Code Tasks

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run GUI",
            "type": "shell",
            "command": "${command:python.interpreterPath}",
            "args": ["run_gui.py"],
            "group": { "kind": "build", "isDefault": true }
        },
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "${command:python.interpreterPath}",
            "args": ["-m", "pytest", "tests/", "-v"],
            "group": { "kind": "test", "isDefault": true }
        },
        {
            "label": "Format Code",
            "type": "shell",
            "command": "${command:python.interpreterPath}",
            "args": ["-m", "black", "src/", "tests/"]
        }
    ]
}
```

#### Launch Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run GUI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/run_gui.py",
            "console": "integratedTerminal"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

### PyCharm

**Settings to Configure:**

1. Project Interpreter
   - Settings → Project → Python Interpreter
   - Select venv folder

2. Code Style
   - Settings → Editor → Code Style → Python
   - Set to match Black formatting

3. Run Configurations
   - Add configuration for `run_gui.py`
   - Add pytest configuration for tests

---

## Git Workflow

### Branch Naming Convention

```
main              - Production branch
develop           - Development branch
feature/my-feature - Feature branches
bugfix/my-fix     - Bug fix branches
docs/my-docs      - Documentation branches
```

### Creating a Feature Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-feature

# Make changes, commit
git add .
git commit -m "Add my feature"

# Push and create PR
git push origin feature/my-feature
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Test additions/changes

**Example:**
```
feat: Add digit recognition confidence visualization

- Add confidence bar chart to prediction display
- Update PredictionDisplay widget to show all digits
- Add color coding for highest confidence

Closes #42
```

---

## Code Style Guidelines

### Format Code with Black

```bash
black src/ tests/
```

### Lint with Flake8

```bash
flake8 src/ tests/
```

**Configuration:** Create `.flake8`:

```ini
[flake8]
max-line-length = 100
exclude = .git,__pycache__,.venv,venv
ignore = E203,W503
```

### Type Checking with Mypy

```bash
mypy src/
```

**Configuration:** Create `mypy.ini`:

```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
```

---

## Testing

### Running Tests

**All tests:**
```bash
pytest tests/
```

**Specific test file:**
```bash
pytest tests/test_model.py -v
```

**Specific test:**
```bash
pytest tests/test_model.py::test_build_cnn_model -v
```

**With coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

**Test file structure:**

```python
# tests/test_my_feature.py
import pytest
from src.my_module import my_function

class TestMyFeature:
    def setup_method(self):
        """Setup before each test"""
        self.test_data = [1, 2, 3]
    
    def test_my_function_success(self):
        """Test successful case"""
        result = my_function(self.test_data)
        assert result == expected_value
    
    def test_my_function_error(self):
        """Test error case"""
        with pytest.raises(ValueError):
            my_function(invalid_data)
```

### Test Coverage

Generate coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

View report: `htmlcov/index.html`

**Coverage goals:**
- Core modules: 80%+
- GUI: 60%+ (harder to test)
- Overall: 70%+

---

## Documentation

### Building Documentation

Documentation is in Markdown format in the `wiki/` directory.

**To edit:**
1. Edit files in `wiki/` directory
2. Use Markdown syntax
3. Follow existing format
4. Test links work

### Markdown Best Practices

```markdown
# Main Heading (H1)

## Section Heading (H2)

### Subsection (H3)

**Bold text** and *italic text*

- Bullet point
- Another point
  - Sub-point

1. Numbered item
2. Another numbered item

[Link text](../path/to/file.md)

`code snippet`

\`\`\`python
# Code block
def my_function():
    pass
\`\`\`

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

---

## Debugging

### Debug Output

Run with debug version:

```bash
python run_gui_debug.py
```

Add debug prints:

```python
print(f"[DEBUG] Variable value: {variable}")
print(f"[DEBUG] Entering function: {function_name}")
```

### Python Debugger (pdb)

```python
import pdb

# Set breakpoint
pdb.set_trace()

# Or in Python 3.7+
breakpoint()
```

**Commands:**
- `n` - Next line
- `s` - Step into
- `c` - Continue
- `p variable` - Print variable
- `l` - List code
- `h` - Help

### VS Code Debugger

Set breakpoints by clicking line numbers. Press F5 to start debugging.

---

## Performance Profiling

### Profile with cProfile

```python
import cProfile

cProfile.run('app_function()')
```

Or command line:

```bash
python -m cProfile -s cumulative run_gui.py
```

### Memory Profiling

```bash
pip install memory-profiler
python -m memory_profiler run_gui.py
```

Or in code:

```python
from memory_profiler import profile

@profile
def my_function():
    # This function will be profiled
    pass
```

---

## Common Development Tasks

### Adding a New Feature

1. Create feature branch: `git checkout -b feature/my-feature`
2. Write failing tests: `tests/test_my_feature.py`
3. Implement feature: `src/my_module.py`
4. Run tests: `pytest tests/test_my_feature.py`
5. Format code: `black src/`
6. Commit: `git commit -m "feat: Add my feature"`
7. Push: `git push origin feature/my-feature`
8. Create pull request on GitHub

### Modifying GUI

1. Edit `src/gui.py` or `src/widgets.py`
2. Test changes: `python run_gui.py`
3. Check layout looks right
4. Run tests: `pytest tests/`
5. Format: `black src/`
6. Commit and push

### Training New Models

1. Run training script:
```bash
python create_models.py
```

2. Or programmatically:
```python
from src.model_ensemble import ModelEnsemble
ensemble = ModelEnsemble()
ensemble.create_new_model(dataset_type='mnist')
```

### Adding Dependencies

1. Install package: `pip install new-package`
2. Add to `requirements.txt`
3. Test: `python run_gui.py`
4. Commit: `git commit -m "Add new-package dependency"`

---

## Troubleshooting Development Issues

### "ImportError: No module named 'src'"

```bash
# Ensure you're in project root
pwd  # Should show CNN-Digit-Recognizer

# Add to path in code
import sys
sys.path.insert(0, '.')
```

### "ModuleNotFoundError in tests"

```bash
# Run from project root
pytest tests/

# Or run specific test
python -m pytest tests/test_model.py
```

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Tests Failing

```bash
# Run in verbose mode
pytest tests/ -vv -s

# Run with print output
pytest tests/ -s

# Run with debugging
pytest tests/ --pdb
```

---

## Release Process

### Creating a Release

1. **Update version:**
   - Update version string in code
   - Update CHANGELOG.md

2. **Create release branch:**
   ```bash
   git checkout -b release/v1.0.0
   ```

3. **Test everything:**
   ```bash
   pytest tests/
   python run_gui.py
   ```

4. **Create tag:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   ```

5. **Push to GitHub:**
   ```bash
   git push origin release/v1.0.0
   git push origin v1.0.0
   ```

6. **Create GitHub release** with notes

---

## Additional Resources

### Python Development
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [Python Documentation](https://docs.python.org/3/)

### Deep Learning
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Keras API](https://keras.io/api/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

### Tools & Frameworks
- [Git Documentation](https://git-scm.com/doc)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)

---

See also: [Architecture Overview](Architecture.md), [Core Modules](Core-Modules.md), [Contributing Guide](Home.md)
