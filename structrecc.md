src/
├── commands/                   # Entry points for all 46+ "/" commands
│   ├── auth.js                 # /login, /logout, /passes, /upgrade
│   ├── context.js              # /clear, /compact, /context, /rewind, /memory
│   ├── help.js                 # /help, /release-notes, /mobile, /feedback
│   ├── integrations.js         # /mcp, /chrome, /skills, /plugin, /hooks, /ide
│   ├── project.js              # /init, /add-dir, /rename, /todos
│   ├── remote.js               # /remote-env, /install-github-app, /install-slack-app
│   ├── session.js              # /exit, /resume, /status, /stats, /usage, /tasks, /sandbox
│   ├── settings.js             # /config, /theme, /output-style, /privacy-settings
│   ├── vcs.js                  # /review, /pr-comments, /fork, /export
│   └── agents.js               # /agents (Manage specialized sub-agent configs)
│
├── components/                 # Ink-based UI Components (React 19)
│   ├── common/                 # UI Primitives
│   │   ├── Divider.js          # createDivider(), createTitledDivider()
│   │   ├── Modal.js            # ModalShell(), AgentCard()
│   │   ├── Spinner.js          # Spinner()
│   │   └── Status.js           # StatusIndicator(), Badge()
│   ├── permissions/            # Interactive permission handlers
│   │   ├── BashPrompt.js       # BashCommandConfirmationPrompt()
│   │   ├── FilePrompt.js       # FileChangePermissionPrompt()
│   │   ├── ToolPrompt.js       # GenericToolConfirmationPrompt()
│   │   └── Explainer.js        # PermissionExplainer() (Ctrl+E logic)
│   ├── terminal/               # Core REPL visualization
│   │   ├── Banner.js           # WelcomeScreen(), AuthenticatedMessageDisplay()
│   │   ├── ContextGrid.js      # ContextVisualizationGrid() (/context grid)
│   │   ├── UsageBar.js         # UsageDisplay(), ExtraUsageDisplay()
│   │   └── SelectInput.js      # useSelectState(), ModelPicker()
│   └── Agent/                  # Sub-agent visualization
│       ├── AgentCard.js        # AgentCard(), AgentCardSimple()
│       ├── AgentWizard.js      # ToolSelectionWizard(), ModelSelectionStep()
│       └── TeammateList.js     # TeamMemberManagementDialog()
│
├── hooks/                      # Stateful Logic & Event Handlers
│   ├── useAutocomplete.js      # getCompletions(), analyzeCompletionContext()
│   ├── useHistory.js           # getBashHistory(), useArrowKeyHistory()
│   ├── useInputHandler.js      # useExitOnCtrlCD(), handleInput()
│   ├── useLsp.js               # useLspDiagnostics(), registerPassiveDiagnostics()
│   ├── useNotifications.js     # useNotifyAfterTimeout(), handleImagePaste()
│   ├── useVimMode.js           # useVimInput(), parseEscapeSequence()
│   └── useStickyBucket.js      # evaluateStickyBucketing()
│
├── services/                   # Protocol Managers & Orchestration
│   ├── auth/                   # Authentication & Security
│   │   ├── ApiKeyManager.js    # validateApiKey(), saveToKeychain()
│   │   ├── OAuthService.js     # exchangeCodeForTokens(), refreshTokens()
│   │   └── KeychainManager.js  # initializeKeychainManager(), readKeychain()
│   ├── lsp/                    # Language Server Protocol
│   │   ├── LspClient.js        # createLspClient(), startLspServer()
│   │   └── LspDiagnostics.js   # normalizeLspDiagnostics()
│   ├── mcp/                    # Model Context Protocol
│   │   ├── ClientManager.js    # connectToServer(), getEnabledPluginIds()
│   │   ├── ToolRegistry.js     # aggregateMcpServers(), resolveToolFromCommand()
│   │   ├── Transports.js       # StdioClientTransport(), WebSocketTransport()
│   │   └── McpContent.js       # processMcpContent() (Blob conversion)
│   └── telemetry/              # Analytics & Monitoring
│       ├── EventLogger.js      # trackTelemetry(), initializeEventLogging()
│       ├── Metrics.js          # MetricsExporter(), calculateDiffStats()
│       └── Tracing.js          # startInteractionSpan(), endLLMRequestSpan()
│
├── tools/                      # LLM Tool Definitions (Prompt Engineering)
│   ├── BashTool/               # prompt.js, validator.js, utils.js
│   ├── FileReadTool/           # prompt.js, reader.js
│   ├── FileWriteTool/          # prompt.js, diff.js
│   ├── SearchTool/             # prompt.js, ripgrep.js
│   └── SubagentTool/           # prompt.js, AgentState.js
│
└── utils/                      # Low-level logic & System Shims
    ├── fs/                     # Filesystem
    │   ├── paths.js            # getWslMountPoint(), isAgentMemoryPath()
    │   ├── safety.js           # isRelativePathSafe(), validatePath()
    │   └── cache.js            # FileCache (LRU), TruncateBuffer
    ├── platform/               # Environment Detection
    │   ├── detector.js         # getPlatform(), isWSL(), isContainerEnv()
    │   └── shell.js            # detectShell(), getGitBashPath()
    ├── security/               # Validation & Sandboxing
    │   ├── bashScanner.js      # validateCommandSecurity(), checkObfuscatedFlags()
    │   ├── sedValidator.js     # validateSedCommand(), validateSedReadOnly()
    │   └── seccomp.js          # findSeccompBpfPath(), applySeccomp()
    └── text/                   # String Processing
        ├── markdown.js         # TerminalMarkdown Lexer/Parser
        ├── width.js            # calculateStringWidth() (p7 - ZWJ/Emoji aware)
        └── tokens.js           # countTokens(), formatTokenStats()