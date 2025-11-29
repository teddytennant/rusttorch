# Project Information

## Author
- **Name**: Theodore Tennant
- **GitHub**: @teddytennant
- **Email**: teddytennant@icloud.com

## Git Configuration
When making commits, always use:
```bash
git config user.name "Theodore Tennant"
git config user.email "teddytennant@icloud.com"
```

## Testing

Use our test class and test runner:

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    ...

if __name__ == "__main__":
    run_tests()
```

To test Tensor equality, use assertEqual.
