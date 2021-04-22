import os
import sys
import requests
import shutil
import glob
import pkg_resources
import yaml

PACKAGE = 'activitysim'
EXAMPLES_DIR = 'examples'
MANIFEST = 'example_manifest.yaml'


def _example_path(resource):
    resource = os.path.join(EXAMPLES_DIR, resource)
    path = pkg_resources.resource_filename(PACKAGE, resource)

    return path


def _load_manifest():
    with open(_example_path(MANIFEST), 'r') as f:
        manifest = yaml.safe_load(f.read())

    assert manifest, f'error: could not load {MANIFEST}'
    return {example['name']: example for example in manifest}


EXAMPLES = _load_manifest()


def add_create_args(parser):
    """Create command args
    """
    create_group = parser.add_mutually_exclusive_group(required=True)
    create_group.add_argument('-l', '--list',
                              action='store_true',
                              help='list available example directories and exit')
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

        get_example(args.example, args.destination)
        sys.exit(0)


def list_examples():
    print('*** Available examples ***\n')

    ret = []
    for example in list(EXAMPLES.values()):
        del example['include']
        ret.append(example)
        print(yaml.dump(example))

    return ret


def get_example(example_name, destination):
    """
    Copy project data to user-specified directory.

    Examples and their data are described in a manifest
    YAML file. Each example contains at least a `name` and
    `include` field which is a list of files/folders to include
    in the copied example.



    Parameters
    ----------

    example_name: str, name of the example to copy.
        Options can be found via list_examples()
    destination: name of target directory to copy files to.
        If the target directory already exists, project files
        will be copied into a subdirectory with the same name
        as the example
    """
    if example_name not in EXAMPLES:
        sys.exit(f"error: could not find example '{example_name}'")

    if os.path.isdir(destination):
        dest_path = os.path.join(destination, example_name)
    else:
        dest_path = destination

    example = EXAMPLES[example_name]

    for item in example.get('include', []):

        # split include string into source/destination paths
        items = item.split()
        assets = items[0]
        if len(items) == 2:
            target_path = os.path.join(dest_path, items[-1])
        else:
            target_path = dest_path

        if assets.startswith('http'):
            download_asset(assets, target_path)

        else:
            for asset_path in glob.glob(_example_path(assets)):
                copy_asset(asset_path, target_path)

    print(f'copied! new project files are in {os.path.abspath(dest_path)}')

    instructions = example.get('instructions')
    if instructions:
        print(instructions)


def copy_asset(asset_path, target_path):

    print(f'copying {os.path.basename(asset_path)} ...')
    if os.path.isdir(asset_path):
        target_path = os.path.join(target_path, os.path.basename(asset_path))
        shutil.copytree(asset_path, target_path)

    else:
        shutil.copy(asset_path, target_path)


def download_asset(url, target_path):

    print(f'downloading {os.path.basename(target_path)} ...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=None):
                f.write(chunk)
