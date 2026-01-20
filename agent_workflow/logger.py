"""Enhanced logging module for MARBLE with level-based control."""

import os
import sys
from typing import Optional


class SimpleLogger:
    """Level-based logger with environment control."""

    def __init__(self):
        """Initialize logger with environment-based control."""
        self.enabled = os.getenv("ENABLE_LOGS", "true").lower() == "true"
        self.verbose = os.getenv("VERBOSE_LOGS", "false").lower() == "true"

        # Detect compile mode to suppress verbose output
        self.compile_mode = "compile" in " ".join(sys.argv)
        if self.compile_mode:
            self.verbose = False  # Force disable verbose in compile mode

        # Log level: ERROR (0), INFO (1), DEBUG (2)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level == "ERROR":
            self.level = 0
        elif log_level == "DEBUG":
            self.level = 2
        else:  # INFO
            self.level = 1

    def debug(self, message: str) -> None:
        """Log debug messages."""
        if self.enabled and self.level >= 2:
            if self.verbose:
                print(f"[DEBUG] {message}", file=sys.stderr)

    def info(self, message: str) -> None:
        """Log informational messages."""
        if self.enabled and self.level >= 1:
            print(f"[INFO] {message}", file=sys.stderr)

    def system(self, message: str) -> None:
        """Log system-level messages."""
        if self.enabled and self.level >= 1:
            print(f"[SYSTEM] {message}", file=sys.stderr)

    def mcp(self, message: str) -> None:
        """Log MCP server status."""
        if self.compile_mode:
            # In compile mode, only show essential MCP messages
            if "✅" in message and ("initialization complete" in message.lower() or "connected" in message.lower()):
                print(f"[MCP] {message}", file=sys.stderr)
        elif self.enabled and self.level >= 2 and self.verbose:
            print(f"[MCP] {message}", file=sys.stderr)
        elif self.enabled and self.level >= 1 and "✅" in message:
            # Only show successful connections at INFO level
            print(f"[MCP] {message}", file=sys.stderr)

    def nodes(self, node_list: list) -> None:
        """Log active nodes list."""
        if self.enabled and self.level >= 2:
            print(f"[NODES] {', '.join(node_list)}", file=sys.stderr)

    def success(self, message: str) -> None:
        """Log success messages."""
        if self.enabled and self.level >= 1:
            print(f"[SUCCESS] {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        """Log warning messages."""
        if self.enabled and self.level >= 1:
            print(f"[WARNING] {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Log errors (always enabled)."""
        # Errors are always printed regardless of settings
        print(f"[ERROR] {message}", file=sys.stderr)


# Global logger instance
logger = SimpleLogger()