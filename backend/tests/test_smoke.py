def test_core_modules_import_without_torch():
    """Importing the CPU surface must never pull torch (Global Constraint).

    Run in a FRESH subprocess: the constraint is about what a clean `import backend.config`
    pulls in, so it must be independent of whatever other tests (e.g. the attribution kernel
    tests) have already imported torch into this session's sys.modules."""
    import subprocess
    import sys

    code = (
        "import sys, backend.schema, backend.events, backend.config\n"
        "assert backend.schema.SCHEMA_VERSION == '1.0'\n"
        "assert 'torch' not in sys.modules, 'torch was pulled in by the CPU surface'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
