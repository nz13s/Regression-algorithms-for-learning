import unittest


class AllTests:
    loader = unittest.TestLoader()
    start_dir = "."
    suite = loader.discover(start_dir, pattern="*Test.py")


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(AllTests.suite)
