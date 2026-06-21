def test_core_modules_import_without_torch():
    """Importing the CPU surface must never pull torch (Global Constraint)."""
    import sys
    import backend.schema
    import backend.events
    import backend.config

    assert backend.schema.SCHEMA_VERSION == "1.0"
    assert "torch" not in sys.modules
