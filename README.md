# Relay Fuzzer (WIP)

Dependencies:
* [TVM](https://github.com/apache/tvm)
* [Python-MIP](https://www.python-mip.com/) (I don't know if it's the best solver, but it can be installed with `pip3 install mip`)

To run the tests, add src to your `$PYTHONPATH`. You can see [the test generators](test/shared_test_generators.py) for basic versions of fuzzers, though the classes in [src](src) are intended to be parameterized to allow for better generation strategies. Note that the test generator can be configured (see the description for the `validate_config` method in the same file).
