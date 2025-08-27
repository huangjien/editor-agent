# Versioning Guide

This project uses [bump2version](https://github.com/c4urself/bump2version) for automated version management with semantic versioning and conventional commits.

## Conventional Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for clear and consistent commit messages:

### Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- **feat**: A new feature (triggers minor version bump)
- **fix**: A bug fix (triggers patch version bump)
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools and libraries
- **ci**: Changes to CI configuration files and scripts

### Breaking Changes
- Add `BREAKING CHANGE:` in the footer or `!` after the type/scope to indicate breaking changes (triggers major version bump)
- Example: `feat!: remove deprecated API endpoint`

### Examples
```bash
# Feature addition (minor bump)
feat(auth): add JWT token validation

# Bug fix (patch bump)
fix(api): handle null response in user endpoint

# Breaking change (major bump)
feat!: redesign authentication system

BREAKING CHANGE: The authentication API has been completely redesigned.
Users must update their integration code.

# Documentation update (no version bump)
docs: update API documentation for v2 endpoints

# Chore (no version bump)
chore: update dependencies to latest versions
```

## Version Bumping

### Automatic Bumping
Use bump2version to automatically update version numbers:

```bash
# Patch version (0.1.2 → 0.1.3)
bump2version patch

# Minor version (0.1.2 → 0.2.0)
bump2version minor

# Major version (0.1.2 → 1.0.0)
bump2version major
```

### What happens automatically:
1. Updates version in `pyproject.toml`
2. Creates a git commit with the version change
3. Creates a git tag (e.g., `v0.1.3`)
4. Uses conventional commit message format

### Manual Process
If you need to bump versions manually:

1. **Update pyproject.toml**: Change the version number
2. **Commit changes**: Use conventional commit message
3. **Create tag**: `git tag v<new-version>`
4. **Push**: `git push && git push --tags`

## Release Workflow

### Development Workflow
1. Create feature branch: `git checkout -b feat/new-feature`
2. Make changes and commit with conventional messages
3. Create pull request
4. After merge to main, bump version appropriately
5. Push tags to trigger release processes

### Version Strategy
- **Patch (0.0.X)**: Bug fixes, small improvements
- **Minor (0.X.0)**: New features, backward-compatible changes
- **Major (X.0.0)**: Breaking changes, major redesigns

### Pre-release Versions
For pre-release versions, you can use:
```bash
# Create alpha/beta/rc versions
bump2version --new-version 0.2.0a1 patch  # Alpha
bump2version --new-version 0.2.0b1 patch  # Beta
bump2version --new-version 0.2.0rc1 patch # Release candidate
```

## Configuration

The versioning configuration is stored in `.bumpversion.cfg`:

```ini
[bumpversion]
current_version = 0.1.2
commit = True
tag = True
tag_name = v{new_version}
message = Bump version: {current_version} → {new_version}

[bumpversion:file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"
```

## Best Practices

1. **Always use conventional commits** for clear change tracking
2. **Bump versions after merging to main** to maintain clean history
3. **Use descriptive commit messages** that explain the "why" not just the "what"
4. **Test before bumping major versions** to ensure backward compatibility is properly handled
5. **Document breaking changes** thoroughly in commit messages and release notes
6. **Keep CHANGELOG.md updated** with significant changes (can be automated later)

## Troubleshooting

### Common Issues

**Version mismatch**: If bump2version fails due to version mismatch:
```bash
# Check current version in files
grep -r "version =" pyproject.toml

# Update .bumpversion.cfg current_version to match
# Then run bump2version again
```

**Uncommitted changes**: Ensure all changes are committed before bumping:
```bash
git status
git add .
git commit -m "chore: prepare for version bump"
bump2version patch
```

**Tag conflicts**: If tag already exists:
```bash
# Delete local tag
git tag -d v0.1.3

# Delete remote tag (if needed)
git push origin :refs/tags/v0.1.3

# Then bump version again
```