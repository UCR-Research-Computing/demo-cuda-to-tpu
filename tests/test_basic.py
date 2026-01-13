def test_demo_runner_import():
    """Simple test to ensure the module can be imported."""
    try:
        import demo_cuda_to_tpu.demo_runner  # noqa: F401

        assert True
    except ImportError:
        assert False, "Failed to import demo_runner from demo_cuda_to_tpu"
