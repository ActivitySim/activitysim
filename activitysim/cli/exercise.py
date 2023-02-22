import os


def add_exercise_args(parser):
    """Create command args"""
    parser.add_argument(
        "example_name",
        type=str,
        metavar="EXAMPLE_NAME",
        help="name of registered external example to test",
    )


def main(args):
    """
    Run tests on a registered external example.
    """
    example_name = args.example_name
    try:
        resultcode = _main(example_name)
    except Exception:
        return 99
    return resultcode


def _main(example_name: str):
    if not example_name:
        print("no example_name given")
        return 101

    import tempfile

    from activitysim.examples.external import exercise_external_example

    tempdir = tempfile.TemporaryDirectory()
    os.chdir(tempdir.name)
    resultcode = exercise_external_example(example_name, tempdir.name)
    return resultcode
