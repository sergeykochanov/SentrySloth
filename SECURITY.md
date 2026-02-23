# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in SentrySloth, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Use one of these private channels:

1. GitHub private vulnerability report:
   `https://github.com/sergeykochanov/SentrySloth/security/advisories/new`
2. Maintainer contact email from GitHub profile (fallback if advisories are unavailable).

Include:

1. A description of the vulnerability
2. Steps to reproduce (if applicable)
3. Potential impact assessment
4. Suggested fix (if you have one)

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 7 days
- **Fix or mitigation**: best effort, typically within 30 days for confirmed issues

## Scope

The following are in scope:

- Prompt injection vulnerabilities that bypass diff sanitization
- Path traversal or arbitrary file read via agentic tool calls
- SQL injection in the cache layer
- Secret leakage (API keys in logs, tracebacks, reports)
- Dependency vulnerabilities in direct dependencies

## Out of Scope

- Vulnerabilities in analyzed repositories (these are expected input)
- LLM hallucinations or false positives/negatives in findings
- Denial of service via large diffs (rate limiting is the user's responsibility)
