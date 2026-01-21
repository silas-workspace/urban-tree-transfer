# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features that have been added but not yet released

### Changed
- Changes in existing functionality

### Deprecated
- Features that will be removed in upcoming releases

### Removed
- Features that have been removed

### Fixed
- Bug fixes

### Security
- Security improvements or vulnerability fixes

## [0.1.0] - YYYY-MM-DD

### Added
- Initial project setup
- Basic project structure
- Core functionality implementation
- Development environment configuration (UV, Nox, Ruff, Pyright)
- Test framework setup (pytest, coverage)
- Documentation (README.md, CLAUDE.md)

---

## Guidelines for Maintaining This Changelog

### Version Format
- Use [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH
  - **MAJOR**: Incompatible API changes
  - **MINOR**: Backwards-compatible new features
  - **PATCH**: Backwards-compatible bug fixes

### Categories
Use these standard categories in order:
1. **Added** - New features
2. **Changed** - Changes in existing functionality
3. **Deprecated** - Soon-to-be removed features
4. **Removed** - Removed features
5. **Fixed** - Bug fixes
6. **Security** - Security-related changes

### Writing Entries
- Write entries in **present tense** (e.g., "Add feature" not "Added feature")
- Start each entry with a **verb** (Add, Change, Fix, Remove, etc.)
- Be **specific** and **concise**
- Include **issue/PR references** when applicable: `(#123)`
- Group related changes together
- Keep the audience in mind (users, not developers)

### Examples

#### Good Entries ✅
```markdown
### Added
- Add user authentication with JWT tokens (#45)
- Add support for PostgreSQL database backend
- Add comprehensive API documentation with examples

### Fixed
- Fix memory leak in data processing pipeline (#67)
- Fix incorrect calculation in statistics module (#72)
```

#### Bad Entries ❌
```markdown
### Added
- Added stuff
- Various improvements
- Updated code

### Fixed
- Fixed bug (too vague)
- Refactored everything (not user-facing)
```

### When to Update
- Update `[Unreleased]` section as you develop
- Create a new version section when releasing
- Move items from `[Unreleased]` to the new version
- Update the version links at the bottom

### Version Links (Optional)
If using Git tags, add comparison links at the bottom:
```markdown
[unreleased]: https://github.com/username/repo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/username/repo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/username/repo/releases/tag/v0.1.0
```

---

## Template for New Releases

When releasing a new version, copy this template:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- 

### Changed
- 

### Deprecated
- 

### Removed
- 

### Fixed
- 

### Security
- 
```

Remove empty sections before publishing.
