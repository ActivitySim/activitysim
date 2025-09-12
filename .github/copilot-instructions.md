# ActivitySim Development Instructions

ActivitySim is an open-source, Python-based activity-based travel behavior modeling software. Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Bootstrap and Setup

**CRITICAL**: Always run setup commands with appropriate timeouts. NEVER CANCEL long-running operations.

- Install uv package manager (required): `pip install uv` (takes ~10 seconds)
- Setup development environment: `uv sync` -- takes 4-5 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
- Python 3.10+ is required (check with `python --version`)
- The project uses `pyproject.toml` for dependency management

## Working Effectively

### Build and Test Commands
- **Core tests**: `uv run pytest --pyargs activitysim.core` -- takes ~18 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- **ABM model tests**: `uv run pytest --pyargs activitysim.abm.models` -- takes ~6 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- **CLI tests**: `uv run pytest --pyargs activitysim.cli` -- takes ~15 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- **Example model tests**: `uv run pytest activitysim/examples/placeholder_multiple_zone/test --durations=0` -- takes 7-8 minutes. NEVER CANCEL. Set timeout to 15+ minutes.

### Linting and Code Quality
- **Black formatting**: `uv run black --check --diff .` -- takes ~7 seconds. Always run before committing.
- **Ruff linting**: `uv run ruff check .` -- takes ~1 second. Note: Shows many existing issues, don't fix unrelated problems.
- **Format with Black**: `uv run black .` -- takes ~10 seconds to format all files.

### CLI Usage
All CLI commands use the pattern: `uv run activitysim <command>`

- **Help**: `uv run activitysim --help` -- takes ~3 seconds
- **List examples**: `uv run activitysim create --list` -- takes ~3 seconds
- **Create example**: `uv run activitysim create -e <example_name> -d <destination>` -- takes 5-10 seconds for small examples
- **Run model**: `uv run activitysim run -c <configs_dir> -d <data_dir> -o <output_dir>` -- varies widely by example size

## Documentation

- **Build docs**: In `docs/` directory, run `make clean && make html` -- takes ~2 minutes. NEVER CANCEL. Set timeout to 5+ minutes.
- **Documentation lives in**: `docs/` directory using Sphinx
- **Built docs output**: `docs/_build/html/`

## Validation and Testing Scenarios

### Essential Test Data Setup
Before running full examples, generate required test data:
```bash
uv run python activitysim/examples/placeholder_multiple_zone/scripts/two_zone_example_data.py
uv run python activitysim/examples/placeholder_multiple_zone/scripts/three_zone_example_data.py
```

### Manual Validation Steps
1. **Always test core functionality**: Run core test suite to verify basic operations
2. **Test CLI commands**: Verify `--help`, `create --list`, and example creation work
3. **Run small example**: Use `placeholder_multiple_zone` tests for comprehensive validation
4. **Validate linting**: Ensure black formatting passes (ruff may show existing issues)

### Example Scenarios Available
Use `uv run activitysim create --list` to see all available examples including:
- `placeholder_2_zone` - 2-zone system test (smallest/fastest)
- `placeholder_multiple_zone` - Multi-zone test scenarios
- `prototype_mtc` - 25-zone MTC prototype
- `prototype_mtc_extended` - Extended MTC example
- Various regional models (PSRC, SANDAG, SEMCOG, ARC, MWCOG)

## Common Tasks and File Locations

### Key Directory Structure
```
activitysim/
├── activitysim/           # Main source code
│   ├── abm/              # Agent-based model components
│   ├── cli/              # Command-line interface
│   ├── core/             # Core functionality
│   ├── estimation/       # Model estimation tools
│   └── examples/         # Built-in example models
├── docs/                 # Documentation source
├── test/                 # Additional test files
└── pyproject.toml        # Project configuration
```

### Important Files to Check
- `pyproject.toml` - Dependencies, tool configuration, and project metadata
- `HOW_TO_RELEASE.md` - Release process and validation steps
- `.github/workflows/core_tests.yml` - CI configuration and test matrix
- `activitysim/examples/` - Contains all example models and test scenarios

### Build Artifacts and Cache
- Always add `test_example/` to `.gitignore` when creating test examples
- Build artifacts go in `_build/` directories
- Cache directories: `**/__sharrowcache__`, `**/skims.zarr`
- Output directories: `**/output/`

## Expected Timing Reference
Use these timings for setting appropriate timeouts:

| Operation | Expected Time | Recommended Timeout |
|-----------|---------------|-------------------|
| `uv sync` | 4-5 minutes | 10+ minutes |
| Core tests | ~18 seconds | 2+ minutes |
| ABM tests | ~6 seconds | 2+ minutes |
| Example tests | 7-8 minutes | 15+ minutes |
| Documentation build | ~2 minutes | 5+ minutes |
| Black linting | ~7 seconds | 1+ minute |
| Ruff linting | ~1 second | 30+ seconds |

## Critical Development Rules
- **NEVER CANCEL** builds, tests, or long-running operations
- Always use `uv run` to execute Python commands in the project environment
- Test both core functionality AND example scenarios when making changes
- Run linting before committing: `uv run black --check --diff .`
- Use `activitysim/examples/placeholder_multiple_zone/test` for comprehensive integration testing
- The project requires Python 3.10+ and uses modern Python features
- Many existing ruff linting issues exist - don't fix unrelated problems

## CI and GitHub Actions
- Primary test workflow: `.github/workflows/core_tests.yml`
- Tests run on Linux, Windows, and macOS
- Regional model testing runs on Windows only
- Documentation builds automatically on main branch
- Use same uv version as CI (0.7.12) for consistency