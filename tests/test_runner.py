import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from backend import runner


class RunnerTests(unittest.TestCase):
    def execute(self, source, expected='', input=''):
        outcome = runner.run_cpp(source, [{'input': input, 'output': expected}])
        self.assertEqual(outcome['compile_status'], 'ok', outcome)
        self.assertEqual(len(outcome['results']), 1)
        return outcome['results'][0]

    def test_compile_once_multiple_cases(self):
        source = '#include <iostream>\nint main(){long a,b;std::cin>>a>>b;std::cout<<a+b;}'
        result = runner.run_cpp(source, [{'input': '19 23', 'output': '42'},
                                         {'input': '-10 4', 'output': '-6'},
                                         {'input': '2 3', 'output': '9'}])
        self.assertEqual(result['compile_status'], 'ok', result)
        self.assertEqual([r['status'] for r in result['results']],
                         ['passed', 'passed', 'wrong_answer'], result)

    def test_compile_error_sanitized(self):
        outcome = runner.run_cpp('int main(){ invalid c++ source }', [{'input': '', 'output': ''}])
        self.assertEqual(outcome['compile_status'], 'compile_error', outcome)
        self.assertIn('<workspace>/main.cpp', outcome['diagnostics'])
        self.assertNotIn('/private/var/folders/', outcome['diagnostics'])
        self.assertEqual(outcome['results'], [])

    def test_private_files_network_writes_and_fork_denied(self):
        # The secret exists outside the per-job directory and contains a unique sentinel.
        with tempfile.TemporaryDirectory(prefix='cppquest-secret-', dir=Path(__file__).resolve().parent) as directory:
            secret = Path(directory) / 'secret.txt'
            secret.write_text('sensitive fixture', encoding='utf-8')
            source = r'''
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cerrno>
#include <cstdlib>
#include <spawn.h>
#include <sys/sysctl.h>
#include <signal.h>
#include <unistd.h>
int main(int argc, char** argv) {
    if(argc>1) return 42;
    std::ifstream private_file(PRIVATE_PATH);
    std::ifstream private_alias(PRIVATE_ALIAS);
    std::ofstream write_attempt("created.txt");
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    addr.sin_port = htons(9); addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int network = connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
    int denied = network == -1 && errno == EPERM;
    pid_t child = fork();
    if (child == 0) _exit(42);
    pid_t spawned; char argument[]="child"; char* args[]={argv[0],argument,nullptr};
    char* child_env[]={nullptr};
    int spawn_result=posix_spawn(&spawned,argv[0],nullptr,nullptr,args,child_env);
    int mib[]={CTL_KERN,KERN_PROCARGS2,PARENT_PID}; char private_args[65536]; size_t size=sizeof(private_args);
    int proc_args=sysctl(mib,3,private_args,&size,nullptr,0);
    std::cout << !private_file.is_open() << !write_attempt.is_open()
              << denied << (child == -1) << !private_alias.is_open()
              << (spawn_result != 0) << (proc_args == -1)
              << (std::getenv("RUNNER_PRIVATE_FIXTURE") == nullptr)
              << (kill(PARENT_PID,0) == -1);
}
'''.replace('PRIVATE_PATH', json.dumps(str(secret))).replace('PRIVATE_ALIAS', json.dumps('/System/Volumes/Data' + str(secret))).replace('PARENT_PID', str(os.getpid()))
            with mock.patch.dict(os.environ, {'RUNNER_PRIVATE_FIXTURE': 'secret environment fixture'}):
                result = self.execute(source, '111111111')
            self.assertTrue(result['passed'], result)

    def test_compile_private_include_denied(self):
        with tempfile.TemporaryDirectory(prefix='cppquest-secret-', dir=Path(__file__).resolve().parent) as directory:
            secret = Path(directory) / 'secret.hpp'
            secret.write_text('int secret=42;', encoding='utf-8')
            source = '#include ' + json.dumps(str(secret)) + '\nint main(){}'
            outcome = runner.run_cpp(source, [{'input': '', 'output': ''}])
            self.assertEqual(outcome['compile_status'], 'compile_error', outcome)
            self.assertIn('Operation not permitted', outcome['diagnostics'])

    def test_infinite_loop_is_killed(self):
        result = self.execute('int main(){ for(;;){} }')
        self.assertEqual(result['status'], 'time_limit', result)
        self.assertLess(result['time_ms'], 3500)

    def test_stdout_flood_is_bounded(self):
        result = self.execute('#include <unistd.h>\nint main(){char x[4096]={};for(;;)write(1,x,sizeof x);}')
        self.assertEqual(result['status'], 'output_limit', result)
        self.assertLessEqual(len(result['stdout'].encode()), runner.OUTPUT_BYTES)

    def test_stderr_flood_is_bounded(self):
        result = self.execute('#include <unistd.h>\nint main(){char x[4096]={};for(;;)write(2,x,sizeof x);}')
        self.assertEqual(result['status'], 'output_limit', result)
        self.assertLessEqual(len(result['stderr'].encode()), runner.OUTPUT_BYTES)

    def test_memory_limit_is_enforced(self):
        source = r'''
#include <cstdlib>
#include <cstring>
#include <unistd.h>
int main(){for(;;){void* p=malloc(16*1024*1024);if(!p)return 3;memset(p,1,16*1024*1024);usleep(2000);}}
'''
        result = self.execute(source)
        self.assertEqual(result['status'], 'memory_limit', result)

    def test_no_fallback_without_sandbox(self):
        runner._toolchain.cache_clear()
        with mock.patch.object(runner, 'SANDBOX', Path('/missing-sandbox')):
            outcome = runner.run_cpp('int main(){}', [{'input': '', 'output': ''}])
        runner._toolchain.cache_clear()
        self.assertEqual(outcome['compile_status'], 'unavailable')

    def test_request_bounds(self):
        with self.assertRaises(ValueError):
            runner.run_cpp('x' * (runner.SOURCE_BYTES + 1), [{'input': '', 'output': ''}])
        with self.assertRaises(ValueError):
            runner.run_cpp('int main(){}', [])
        with self.assertRaises(ValueError):
            runner.run_cpp('int main(){}', [{'input': 'x' * (runner.INPUT_BYTES + 1), 'output': ''}])
        with self.assertRaises(ValueError):
            runner.run_cpp('int main(){}', [{'input': '', 'output': ''}] * 13)

    def test_profile_does_not_grant_runtime_forks_or_writes(self):
        profile = runner._profile(Path('/private/tmp/job'), Path('/Library/Developer/CommandLineTools/usr/bin/clang++'),
                                  Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'),
                                  Path('/Library/Developer/CommandLineTools'), False)
        self.assertIn('(deny default)', profile)
        self.assertNotIn('(allow process-fork)', profile)
        self.assertNotIn('(allow file-write', profile)
        self.assertNotIn('(allow network', profile)


if __name__ == '__main__':
    unittest.main(verbosity=2)
