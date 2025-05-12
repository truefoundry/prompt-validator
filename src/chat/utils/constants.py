from enum import Enum

class RequestType(Enum):
    VALIDATION = "validation"
    VERIFY_TESTS = "verify_tests"
    VERIFY_TESTS_EXACT ="verify_tests_exact"
