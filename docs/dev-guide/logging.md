# Logging

ActivitySim uses the usual Python [logging](https://docs.python.org/3/library/logging.html)
infrastructure, with just a few additional features.

Generally, logging configuration is done via the
[dictConfig](https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig)
interface, with keys and values as documented
[here](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details).
This dictionary fed to this configurator is loaded from the `logging.yaml`
file(s) located in your model's configuration directory(s) following the
usual pattern for finding and loading config files.

```{versionadded} 1.3
ActivitySim no longer permits the use of `!!python/object/apply` directives inside
yaml input files.  These commands imply the capability to allow arbitrary code
execution, and we would like to move away from that.
```

Instead of allowing arbitrary code to be loaded into and modify the logging configuration,
there are just a few particular ActivitySim functions are exposed.

## Log file locations

As noted above, the logging configuration implementation relies heavily on the
standard Python logging library, which by default knows nothing about ActivitySim
or its typical layout of output files, including placement of logs in a designated
output directory.  Therefore, if you set the filename of a logging FileHandler to
just a string like this:

```yaml
logfile:
  class: logging.FileHandler
  filename: just-a-file-name.log
```

then that file will be created in the Python current working directory (typically
wherever you invoked the script) and not in your designated output directory.
To fix this and write the log into your designated output directory, you can use
`get_log_file_path` as an intervening key in the configuration between the
`filename` key and the desired value, like this:

```yaml
logfile:
  class: logging.FileHandler
  filename:
    get_log_file_path: my-file-name.log
```

This special formatting will be pre-processed by ActivitySim before configuring
the logging, so that the file will be created in your designated output directory.
This also works when subprocesses are running, in which case the log file will
then be created in (or relative to) the process' log file directory,
not in (or relative to) the main output directory.

## Identifying Subprocesses

You may want to have different settings for subprocess workers and the main
ActivitySim process.  For example, you may have the main processes log everything it writes
to both the console and a log file, while the subprocesses log mostly to files, and
only write higher priority messages (warnings and errors) to the console.  Any
logging configuration can be set to bifurcate like this between the main process and
subtasks by setting "is_sub_task" and "is_not_sub_task" keys like this:

```yaml
handlers:
  console:
    level:
      if_sub_task: WARNING
      if_not_sub_task: NOTSET
```
