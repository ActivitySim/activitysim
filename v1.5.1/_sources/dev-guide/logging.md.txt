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

## Logging levels
Python's built-in `logging` module that includes five levels of logging, which are (in order
of increasing severity): `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. One can set the
minimum level to display messages in both the console window as well as the output logfile
within `logging.yaml` in the model settings. For example, if the block of code below were
inside the `logging.yaml` file, than the console window and output activitysim.log file would
print every logging message at the level of `INFO` and above:

```yaml
loggers:
  activitysim:
    level: INFO
    handlers: [console, logfile]
    propogate: false
```

However, if a model run were to crash and the user wanted to print all of the `DEBUG` messages
in order to diagnose what was causing the crash, they would need to change the `level` within
the logging settings:

```yaml
loggers:
  activitysim:
    level: DEBUG
    handlers: [console, logfile]
    propogate: false
```

The following guidelines demonstrate how each level is used within ActivitySim:

### Debug (Level 10)
The `DEBUG` message indicates detailed information that would be of interest to a user while
debugging a model. The information reported at this level can include:
- Runtimes of specific steps of model components, such as the time to run each of sampling,
  logsum computation, and simulation in destination choice
- Table attributes at various stages of processing, such as the size or columns
- Evaluations of preprocessor or specification expressions
- General repetitive messages that can be used to narrow down exactly where an error is occuring

### Info (Level 20)
The `INFO` message gives reports general information about how the status of the model run,
particularly where in the model flow the system is at. The information reported at this level
can include:
- Beginning and ending of a model step
- Intermediate stages of a longer step. For example, in trip destination, the trip number and
  segment will be reported at this level.

### Warning (Level 30)
The `WARNING` message notifies the user of a potential issue that they should be aware of,
but doesn't result in the model system failing. The information reported at this level can include:
- Future changes to dependencies
- ActivitySim needing to force certain travel behavior due to such behavior not working

### Error (Level 40)
The `ERROR` message gives the user information that is causing an error in a model step. The
information reported at this level can include:
- More detailed issues on what could be causing an error message that wouldn't be shown in the
  traceback message

## Critical (Level 50)
The `CRITICAL` message gives the user information that is causing a critical error in a model step.
The information reported at this level can include:
- Reporting to the user on the teardown of a subprocess