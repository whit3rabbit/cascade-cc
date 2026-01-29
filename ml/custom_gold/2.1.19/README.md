# Claude Code Source Structure

This directory contains the deobfuscated and refactored source code for Claude Code.

## Architecture Overview

The codebase is organized into modular services and components, moving away from the original bundled structure.

### `src/services/`
- **`terminal/`**: Core control stream client/service, command registry, and shell environment management.
- **`auth/`**: Authentication managers (API keys, OAuth) and token persistence.
- **`mcp/`**: Model Context Protocol integration, server management, and tool processing.
- **`config/`**: User/Project settings and onboarding flows.
- **`agents/`**: Custom agent persistence, validation, and context management.
- **`telemetry/`**: OpenTelemetry initialization and event tracking.
- **`logging/`**: Buffered logging to disk and MCP log management.
- **`sandbox/`**: Permission rules and sandbox configuration for tool execution.

### `src/components/` (Ink/React)
- **`wizard/`**: UI steps for the agent creation wizard.
- **`permissions/`**: Dialogs for tool execution confirmation.
- **`common/`**: Reusable UI elements like `Card`.

### `src/tools/`
- **`ToolGroups.js`**: Defines tool categories (Read, Edit, Execution, MCP).
- Actual tool implementations are being migrated from the legacy `prompt.js` files into structured service calls.

### `src/utils/`
- **`fs/`**: File system utilities (paths, watching, parsing).
- **`http/`**: Network stack, retries, and Node-to-browser shims.
- **`shared/`**: Common data structures, lodash/bash helpers, and runtime validation.
- **`text/`**: ANSI and string processing utilities.

## Key Services
- **`AppInitializer`**: Coordinates the startup sequence of all other services.
- **`ControlStreamClient`**: Manages the persistent connection to the Claude backend.
- **`ToolExecutionManager`**: Orchestrates parallel/serial execution of agent tools.
- **`PermissionService`**: Enforces local sandbox rules and handles user confirmation.
