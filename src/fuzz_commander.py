import subprocess
import time
import base64
import json

import tvm
from tvm import relay

def time_since_epoch():
    # gmtime returns time struct
    # mktime takes time struct and returns float sec since epoch
    int(time.mktime(time.gmtime()))

class FuzzCommander:
    def __init__(self, fuzz_file_dir, output_dir, seed=0):
        """
        
        """
        pass

class FuzzOnceFromFileName:
    def __init__(self, payload_identifier, on_timeout, on_non_zero_exit, timeout_secs=10):
        """
        payload_identifier: This should be some identifier that points to 
                            the test case (fuzzed program) to run. For now,
                            it is a file name (fully qualified file name).
        on_timeout:         A callback to call with self as its argument on
                            timeout, will probably be passed from FuzzCommander
        on_non_zero_exit:   A callback to call with self as its argument on
                            non-zero exits, probably passed from FuzzCommander
        timeout_secs:       The number of seconds the subprocess is allowed to 
                            run before terminating.
        """
        self.payload_identifier = payload_identifier
        self.timeout_secs = timeout_secs
        self.compile_time_start = None
        self.compile_time_end = None
        self.test_runner_shell_cmd = "python3 \
                                      ~/repos/relay_fuzzer/src/test_runner.py \
                                      {}".format(self.payload_identifier)
        self.did_crash = False
        self.did_timeout = False
        self.stdout = ""
        self.stderr = ""
        self.exit_code = 0
        self.problem_timestamp = None # crash/timeout timestamp (secs since epoch)
        # self.did_crash and non-zero exit code are
        # different, TVM might just reject the program
        self.on_timeout = on_timeout
        self.on_non_zero_exit = on_non_zero_exit

    def run(self):
        try:
            # check=True means that if there is a non-zero exit code, raise the
            # CalledProcessError exception and that exception will hold the
            # command (and args), exit code, stderr, and stdout
            # if timeout seconds are reached, then TimeoutExpired will raise
            self.compile_time_start = time.time()
            subprocess.run(self.test_runner_shell_cmd, shell=True,
                           check=True, timeout=self.timeout_secs)
            self.compile_time_end = time.time()
        except subprocess.CalledProcessError as cpe:
            self.compile_time_end = time.time()
            self.exit_code = cpe.returncode
            self.stdout = cpe.stdout
            self.stderr = cpe.stderr
            self.on_non_zero_exit(self)
        except subprocess.TimeoutExpired as te:
            self.did_timeout = True
            self.exit_code = -1 # Some err in case someone checks run's return
            self.on_timeout(self)
        return self.exit_code

    def total_compile_time(self):
        if self.did_timeout:
            return self.timeout_secs
        elif self.compile_time_start and self.compile_time_end:
            return self.compile_time_end - self.compile_time_start
        else:
            return None
        
class CrashJson:
    def __init__(self, identifier="", problem_type="crash", problem_timestamp=0,
                 compile_time=0.0, stdout_str="", stderr_str="", exit_code=-1):
        """
        Json that will be stored on disk in case of crash or timeout
        identifier:        Some pointer to the crash, for now is a filename
        problem_type:      one of ["crash", "timeout"]
        problem_timestamp: time of problem in seconds since epoch (int)
        compile_time:      time taken to compile the program (float)
        stdout_str:        string output of stdout of problematic run, written to
                           json file as base64 encoded string
        stderr_str:        string output of stderr of problematic run, written to
                           json file as base64 encoded string
        exit_code:         exit code from the program (-1 on timeout)
        """
        self.identifier = identifier
        self.problem_type = problem_type
        self.problem_timestamp = problem_timestamp
        self.compile_time = compile_time
        self.stdout = base64.b64encode(stdout_str)
        self.stderr = base64.b64encode(stderr_str)
        self.exit_code = exit_code
        self.internal_dict = {}

    def dumps(self):
        self.internal_dict["identifier"] = self.identifier
        self.internal_dict["problem_type"] = self.problem_type
        self.internal_dict["problem_timestamp"] = self.problem_timestamp
        self.internal_dict["compile_time"] = self.compile_time
        self.internal_dict["stdout"] = self.stdout
        self.internal_dict["stderr"] = self.stderr
        self.internal_dict["exit_code"] = self.exit_code
        return json.dumps(self.internal_dict, indent=4)
        
    def write_to_file(self, filename):
        json_str = self.dumps()
        with open(filename, 'w') as fh:
            fh.write(json_str)

if __name__ == '__main__':
    import sys
    fuzz_once = FuzzOnceFromFileName(sys.argv[1],
                                     (lambda x: print("timeout")),
                                     (lambda x: print("non-zero")))
    fuzz_once.run()
    
