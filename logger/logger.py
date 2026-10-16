from datetime import datetime
import os
from enum import Enum
from string import Template



class SOURCE(Enum):
    DEBUGGER = "DEBUGGER"
    LOGGER = "LOGGER"
    CHECKPOINT = "CHECKPOINT"

class Logger:
    color_code_map = {
        SOURCE.DEBUGGER      : "\033[31m",   #red
        SOURCE.LOGGER        : "\033[32m",   #green
        SOURCE.CHECKPOINT    : "\033[36m",   #cyan
        "BLANK"         : "\033[33m",
    }

    reset = "\033[0m"
    debug_mode = False

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, debug_mode=False, logs_dir_path=""):
        if getattr(self, "_initialized", False):
            return

        Logger.debug_mode = debug_mode

        self.debug_count = 0
        self.log_count = 0
        self.checkpoint_count = 0
        self.rjust_const = 4
        self.logs_dir = logs_dir_path
        os.makedirs(self.logs_dir, exist_ok=True)

        self.debug_log_file = os.path.join(self.logs_dir, "debug.logs")
        self.log_file = os.path.join(self.logs_dir, "logs.logs")
        self.checkpoint_file = os.path.join(self.logs_dir, "checkpoint.logs")
        self.all_logs_file = os.path.join(self.logs_dir, "all.logs")
        # self.template = "{color_code}[{source}:{counter}]{reset} {string} @ {time}{end}"
        self.template = Template(template="$color_code[$source:$counter]$reset $string @ $time$end")
        self._initialized = True

    # def _log_colored_to_console(self, string: str, source, counter, color, end="\n"):
    #     color_code = self.color_code_map[color]
    #     print(f"{color_code}[{source}:{counter}] "f"{self.reset}{string}", end=end,)
    
    
    #? just for the sake of abstraction, but this is total shift obviously
    def _write_to_console(self, string, end="\n"):
        print(string)
    
    
    def _write_to_file(self, file_path, string):
        with open(file_path, "a") as f:
            f.write(f"{string}\n")

    def debug(self, string: str, obj=None, end="\n"):
        if  Logger.debug_mode is not True:
            return
        
        timestamp, color_code, counter_str = self.get_formatting_info(SOURCE.DEBUGGER, self.debug_count)
        
        if obj is not None:
            string = f"{string}: {obj}"

        console_string, file_string = self.format_string(string, timestamp, color_code, counter_str, SOURCE.DEBUGGER, end)
        self._write(self.debug_log_file, console_string, file_string)

        self.debug_count += 1





    def log(self, string: str, obj=None, end="\n"):
        timestamp, color_code, counter_str = self.get_formatting_info(SOURCE.LOGGER, self.log_count)

        if obj is not None:
            string = f"{string}: {obj}"

        console_string, file_string = self.format_string(string, timestamp, color_code, counter_str, SOURCE.LOGGER, end)
        self._write(self.log_file, console_string, file_string)

        self.log_count+=1

    def checkpoint(self, string: str, end="\n"):
        
        
        timestamp, color_code, counter_str = self.get_formatting_info(SOURCE.CHECKPOINT, self.checkpoint_count)
        
        console_string, file_string = self.format_string(string, timestamp, color_code, counter_str, SOURCE.CHECKPOINT, end)
        self._write(self.checkpoint_file, console_string, file_string)

        self.checkpoint_count+=1
        
        

    def _write(self, file, console_string, file_string):
        self._write_to_console(console_string)
        self._write_to_file(file, file_string)
        self._write_to_file(self.all_logs_file, file_string)

    def get_formatting_info(self, source, count):
        timestamp = datetime.now().strftime('%H-%M-%S')
        color_code = self.color_code_map[source]
        counter_str = str(count).rjust(self.rjust_const, "0")
        return timestamp,color_code,counter_str

    def format_string(self, string, timestamp, color_code, counter_str, source, end):
        template = self.template
        template = Template(template.safe_substitute(
            source = source,
            counter = counter_str,
            string = string,
            time = timestamp,
            end=end
        ))
        console_string = template.safe_substitute(
            color_code = color_code,
            reset = self.reset
        )
        file_string = template.safe_substitute(
            color_code = "",
            reset = ""
        )

        return console_string,file_string