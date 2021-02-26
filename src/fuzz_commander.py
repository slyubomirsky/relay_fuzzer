import subprocess
import time
import base64
import json
from pathlib import Path
import errno
import glob
import os

import tvm
from tvm import relay

test_runner = Path("~/repos/relay_fuzzer/src/test_runner.py")

class FuzzCommander:
    def __init__(self, fuzz_file_dir, output_dir, seed=0, heartbeatintrvl=300):
        """
        The FuzzCommander
        fuzz_file_dir: Directory where generated fuzzed files are stored. There
                       should not be anything in fuzz_file_dir except files to
                       be used for fuzzing.
        output_dir:    Directory where crash-info files should be written
        seed:          the seed used to generate the fuzzed files
        """
        self.fuzz_file_dir = Path(fuzz_file_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.start_time = 0
        self.heartbeatintrvl = heartbeatintrvl

        self.raise_on_dir_not_exist(self.fuzz_file_dir)
        self.raise_on_dir_not_exist(self.output_dir)

        self.total_number_of_files = len(os.listdir(self.fuzz_file_dir))
        self.files_fuzzed = 0
        self.last_heartbeat_time = int(time.time())

    def run_all_fuzz_files(self):
        self.start_time = int(time.time())
        for fuzzfile in glob.iglob(str(self.fuzz_file_dir / "*")):
            self.run_one_fuzz_file(fuzzfile)
            self.heartbeat()
        print("Ran out of files. Total time: {}. Total files ran: {}".format(
            int(time.time()) - self.start_time,
            self.files_fuzzed
        ))

    def run_one_fuzz_file(self, filename):
        fuzz_run = FuzzOnceFromFileName(filename,
                                        self.get_problem_callback(),
                                        self.get_problem_callback())
        fuzz_run.run()
        self.files_fuzzed += 1

    def heartbeat(self):
        current_time = int(time.time())
        if current_time - self.last_heartbeat_time > self.heartbeatintrvl:
            self.last_heartbeat_time = current_time
            print("PULSE. Progress ({0:.0%}). Live for {1} minutes.".format(
                self.files_fuzzed / self.total_number_of_files,
                int((current_time - self.start_time) / 60)
            ))

    def get_problem_callback(self):
        def callback(fuzz_run):
            # fuzz_run is an instance of FuzzOnceFromFileName that
            # has exited non-zero or timed out
            crash_json = fuzz_run.to_crash_json()

            # These two lines for testing only, just in case we do find a crash
            # and we accidnetally lose it because the actual file writing isn't
            # working or something errors
            tmpfile = Path("tmp.crash")
            crash_json.write_to_file(str(tmpfile))
            
            output_dir = Path(self.output_dir)
            base_name = Path("{}.crash".format(fuzz_run.payload_identifier))
            write_output_to = output_dir / base_name
            
            crash_alert = "CRASH! From {} after {} sec, writing to {}".format(
                fuzz_run.payload_identifier,
                fuzz_run.problem_timestamp - self.start_time,
                str(write_output_to)
            )
            print(crash_alert)
            crash_json.write_to_file(str(write_output_to))
        return callback

    def raise_on_dir_not_exist(self, d):
        if not d.exists() or not d.is_dir():
             raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                     str(d))

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
        self.test_runner_shell_cmd = "python3 {} {}".format(str(test_runner),
                                                            self.payload_identifier)
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
            self.problem_timestamp = int(time.time())
            self.on_non_zero_exit(self)
        except subprocess.TimeoutExpired as te:
            self.did_timeout = True
            self.exit_code = -1 # Some err in case someone checks run's return
            self.problem_timestamp = int(time.time())
            self.on_timeout(self)
        return self.exit_code

    def total_compile_time(self):
        if self.did_timeout:
            return self.timeout_secs
        elif self.compile_time_start and self.compile_time_end:
            return self.compile_time_end - self.compile_time_start
        else:
            return None

    def to_crash_json(self):
         return CrashJson(self.payload_identifier,
                          "timeout" if self.did_timeout else "crash",
                          self.problem_timestamp,
                          self.total_compile_time(),
                          self.stdout,
                          self.stderr,
                          self.exit_code)
        
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
    
    if len(sys.argv) != 3:
        print("Usage: python3 fuzz_commander.py \
              <path_to_fuzzed_files> \
              <path_to_store_crash_info_in>")
        sys.exit(-1)
        
    fuzz = FuzzCommander(sys.argv[1], sys.argv[2])
    fuzz.run_all_fuzz_files()
    
