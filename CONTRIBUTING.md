# Contributing to AI Outlier Detection Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ai-outlier-detection.git
   cd ai-outlier-detection
   ```

2. **Set up Development Environment**
   ```bash
   make dev-install
   make setup
   ```

3. **Run Tests**
   ```bash
   make test
   ```

## 📋 Development Workflow

### Code Style
- Use **Black** for code formatting: `make format`
- Follow **PEP 8** guidelines
- Use **type hints** for all functions
- Maximum line length: **100 characters**

### Testing
- Write tests for all new features
- Maintain **>90% code coverage**
- Run tests before submitting: `make test`
- Use pytest fixtures for reusable test data

### Linting
- Run linting before commits: `make lint`
- Fix all flake8 warnings
- Ensure mypy type checking passes

## 🔧 Project Structure

```
ai-outlier-detection/
├── src/                    # Core pipeline code
│   ├── config.py          # Configuration management
│   ├── pipeline.py        # Main pipeline orchestration
│   ├── data_preprocessing.py
│   ├── embedding_generator.py
│   ├── outlier_detectors.py
│   ├── explainer.py
│   └── visualizer.py
├── tests/                 # Test suite
├── examples/              # Usage examples
├── cli.py                # Command-line interface
├── api.py                # FastAPI web interface
└── docs/                 # Documentation
```

## 🎯 Contributing Guidelines

### Issues
- Use issue templates when available
- Provide clear reproduction steps for bugs
- Include system information and error messages

### Pull Requests
1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test Your Changes**
   ```bash
   make check  # Runs lint + test
   ```

4. **Submit PR**
   - Use descriptive commit messages
   - Reference related issues
   - Include tests and documentation updates

### Commit Messages
Follow conventional commits format:
```
type(scope): description

feat(detectors): add new anomaly detection algorithm
fix(api): handle empty request bodies
docs(readme): update installation instructions
test(pipeline): add integration tests
```

## 🧪 Testing Guidelines

### Unit Tests
- Test individual components in isolation
- Use mocks for external dependencies
- Cover edge cases and error conditions

### Integration Tests
- Test component interactions
- Use real data when possible
- Test API endpoints end-to-end

### Test Structure
```python
def test_feature_name():
    """Test description."""
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = function_under_test(setup_data)
    
    # Assert
    assert result.expected_property == expected_value
```

## 📚 Documentation

### Code Documentation
- Use docstrings for all public functions
- Include parameter types and return values
- Provide usage examples

### README Updates
- Update README.md for new features
- Include code examples
- Update installation instructions if needed

## 🔒 Security

### API Keys
- Never commit API keys or secrets
- Use environment variables
- Update .env.example for new variables

### Dependencies
- Keep dependencies up to date
- Review security advisories
- Use `pip audit` for vulnerability scanning

## 🚀 Release Process

### Version Numbering
Follow semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Docker image builds successfully

## 💬 Getting Help

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Email**: Contact maintainers for security issues

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to AI Outlier Detection Pipeline! 🎉