"""Provider-neutral local coding-agent runner contracts.

This package defines the input/output models, the ``Runner`` protocol, and the
registry used to select a local runner (e.g. a future Codex subprocess runner)
without importing or executing any vendor implementation. Importing this
package has no side effects and never launches a vendor CLI.
"""
