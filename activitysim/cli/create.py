import os
import sys
import shutil
import pkg_resources

PACKAGE = 'activitysim'
EXAMPLES_DIR = 'examples'


def add_create_args(parser):
    """Create command args
    """
    create_group = parser.add_mutually_exclusive_group(required=True)
    create_group.add_argument('-l', '--list',
                              action='store_true',
                              help='list available example directories')
    create_group.add_argument('-e', '--example',
                              type=str,
                              metavar='PATH',
                              help='example directory to copy')

    parser.add_argument('-d', '--destination',
                        type=str,
                        metavar='PATH',
                        default=os.getcwd(),
                        help="path to new project directory (default: %(default)s)")


def create(args):
    """
    Create a new ActivitySim configuration from an existing template.

    Use the -l flag to view a list of example configurations, then
    copy to your own working directory. These new project files can
    be run with the 'run' command.
    """

    if args.list:

        list_examples()
        sys.exit(0)

    if args.example:

        copy_example(args.example, args.destination)
        sys.exit(0)


def list_examples(example_pardir=EXAMPLES_DIR, package=PACKAGE):
    example_dirs = pkg_resources.resource_listdir(package, example_pardir)
    print('Available examples:')
    for example in example_dirs:
        print("\t"+example)


def copy_example(example,
                 destination,
                 example_pardir=EXAMPLES_DIR,
                 package=PACKAGE):
    example_dirs = pkg_resources.resource_listdir(package, example_pardir)
    if example not in example_dirs:
        sys.exit("error: could not find example '%s'" % example)

    if os.path.isdir(destination):
        dest_path = os.path.join(destination, example)
    else:
        dest_path = destination

    resource = os.path.join(example_pardir, example)
    example_path = pkg_resources.resource_filename(package, resource)

    print('copying files from %s...' % example)
    shutil.copytree(example_path, dest_path)

    print("copied! new project files are in %s" % os.path.abspath(dest_path))
