── commands
│   ├── SlashCommandDispatcher.ts (Class: SlashCommandDispatcher; Handles: /routing, logic fallthrough to LLM)
│   ├── help.tsx (Component: HelpComponent; Commands: /help, /mobile)
│   ├── helpers.ts (Interface: CommandContext, CommandDefinition; Function: createCommandHelper)
│   ├── init.ts (Command: /init; Role: CLAUDE.md and project initialization)
│   ├── pr_comments.ts (Command: /pr-comments; Utility: GitHub CLI wrapper)
│   └── release-notes.ts (Command: /release-notes; Utility: Version update display)
├── components
│   ├── MessageHistory.tsx (Component: MessageHistory; Role: Renders Markdown & Highlighted Code)
│   ├── ModelPicker
│   │   └── ModelPicker.tsx (Component: ModelPicker; Role: Model selection UI)
│   ├── TaskList.tsx (Interface: TaskListProps; Role: Background task list display)
│   ├── common
│   │   ├── ActionHint.tsx (Component: ActionHint; Role: Keybind shortcut display)
│   │   ├── BashOutputComponents.tsx (Components: UserMemoryInput, OutputLine; Role: Shell output rendering)
│   │   ├── Card.tsx (Component: Card; Role: Styled grouping container)
│   │   ├── Divider.tsx (Component: Divider; Role: Horizontal separator)
│   │   ├── Dividers.tsx (Component: TitledDivider; Role: Separators with labels)
│   │   ├── Select.tsx (Interface: Option; Component: Select; Role: Interactive list selector)
│   │   └── StatusDisplay.tsx (Interface: Task, SessionData; Role: Task progress indicators)
│   ├── mcp
│   │   ├── MCPServerDialog.tsx (Role: Marketplace and Plugin management UI)
│   │   └── ReconnectMcpServer.tsx (Component: ReconnectMcpServer; Role: Server lifecycle UI)
│   ├── messages
│   │   └── UserPromptMessage.tsx (Component: UserPromptMessage; Features: Vim mode, History navigation)
│   ├── onboarding
│   │   ├── IdeOnboardingDialog.tsx (Role: IDE setup wizard)
│   │   └── LspRecommendationDialog.tsx (Role: Language Server recommendations)
│   ├── permissions
│   │   ├── ConfirmationDialogs.tsx (Components: ToolConfirmationDialog; Specialized UI for Bash/MCP calls)
│   │   ├── CostThresholdDialog.tsx (Role: Token/Cost limit warnings)
│   │   ├── PermissionDialog.tsx (Role: Main approval modal for tool execution)
│   │   ├── PermissionOption.tsx (Component: PermissionOption; Role: Accept/Reject/Mode buttons)
│   │   ├── SandboxPermissionDialog.tsx (Role: Network/Filesystem sandbox UI)
│   │   ├── WorkerPermissionDialog.tsx (Role: Parallel worker permissions)
│   │   └── usePermissionDialog.ts (Hook: usePermissionDialog; Logic: Permission state/feedback)
│   ├── terminal
│   │   ├── Logo.tsx (Component: Logo; Role: ASCII art and build info)
│   │   ├── REPL.tsx (Component: REPL; Main entry point for the interactive session)
│   │   ├── StatusLine.tsx (Component: StatusLine; Role: Persistent Footer/Vim bar)
│   │   └── TranscriptToggle.tsx (Component: TranscriptToggle; Role: Detail view controls)
│   └── wizard
│       └── AgentWizardSteps.tsx (Components: ToolSelectionStep, ModelSelectionStep; Role: Custom agent builder)
├── constants
│   ├── build.ts (Info: BUILD_INFO; Function: getUserAgent)
│   ├── keys.ts (Keys: Telemetry IDs, Storage keys, Context symbols)
│   ├── product.ts (Constants: PRODUCT_NAME, CLIENT_ID, Beta Headers)
│   └── theme.ts (Colors: THEME; Role: Terminal color palette)
├── entrypoints
│   └── cli.tsx (Function: main; Modes: Interactive REPL vs. Print/Non-interactive)
├── hooks
│   ├── useAutocomplete.tsx (Hook: useAutocomplete; Services: FileIndexService)
│   ├── useHistory.tsx (Hook: useHistory; Services: HistoryService)
│   ├── usePermissionContext.ts (Hook: usePermissionContext; Role: Tool permission state bridge)
│   └── useTerminalInput.ts (Hook: useTerminalInput; Role: Line editing and cursor management)
├── services
│   ├── agents
│   │   ├── AgentPersistence.ts (Interface: AgentData; Functions: saveAgent, deleteAgent, formatAgentMarkdown)
│   │   ├── AgentValidator.ts (Interface: AgentConfig; Function: validateAgentConfig)
│   │   └── TeammateContextService.ts (Class: TeammateContextService; Storage: AsyncLocalStorage for Swarms)
│   ├── anthropic
│   │   └── AnthropicClient.ts (Classes: AnthropicFilesAPI, AnthropicModelsAPI; Role: Low-level API client)
│   ├── auth
│   │   ├── AccountManager.ts (Interface: Account; Class: AccountManager; Role: MSAL account storage)
│   │   ├── ApiKeyManager.ts (Class: ApiKeyManager; Errors: Loopback server errors)
│   │   ├── AuthService.ts (Functions: getAuthHeaders, isAuthenticated)
│   │   ├── KeychainService.ts (Interface: KeychainInterface; Role: macOS Security/Keychain wrapper)
│   │   ├── MsalAuthService.ts (Class: MsalAuthService; Logic: Token refresh/silent auth)
│   │   ├── OAuthService.ts (Class: OAuthService, LoopbackServerHandler; Role: PKCE Auth flow)
│   │   ├── azureCredentials.ts (Classes: AzureCliCredential, AzureDeveloperCliCredential)
│   │   └── msalConfig.ts (Function: createMSALConfiguration; Role: Auth telemetry tracking)
│   ├── config
│   │   ├── ConfigStatusService.ts (Function: getDetailedConfigStatus; Role: Diagnostics)
│   │   ├── RemoteSettingsService.ts (Function: fetchRemoteSettings; Role: Feature flag caching)
│   │   └── SettingsService.ts (Interface: Settings; Function: updateSettings; Role: User/Project config)
│   ├── conversation
│   │   ├── ConversationService.ts (Class: ConversationService; Role: Main startConversation loop)
│   │   └── PromptManager.ts (Class: PromptManager; Role: System prompt assembly & token breakdown)
│   ├── documentation
│   │   └── DocumentationService.ts (Class: DocumentationService; Role: Local docs cache/fetcher)
│   ├── git
│   │   └── RepositoryTracker.ts (Class: RepositoryTracker; Role: Maps repo URLs to local paths)
│   ├── ide
│   │   └── ideDetection.ts (Role: Scans for JetBrains/VSCode Claude plugins)
│   ├── installer
│   │   └── ChromeExtensionInstaller.ts (Role: Native Host manifest installer for Chrome)
│   ├── logging
│   │   └── LogManager.ts (Class: LogManager; Role: JSONL buffered file logging)
│   ├── marketplace
│   │   ├── MarketplaceLoader.ts (Functions: loadMarketplaces; Role: Metadata fetching)
│   │   └── MarketplaceService.ts (Interface: MarketplaceServiceType; Role: Plugin source management)
│   ├── mcp
│   │   ├── McpBundleLoader.ts (Role: Environment expansion for MCP configs)
│   │   ├── McpClientManager.ts (Class: McpClientManager; Role: Active client lifecycle)
│   │   ├── McpConfigManager.ts (Class: McpConfigManager; Role: Temp directory config storage)
│   │   ├── McpConnectionProvider.tsx (Hooks: useMcpReconnect, useMcpToggleEnabled)
│   │   ├── McpContent.ts (Functions: processOutput, parseBalancedJSON; Role: Tool output parsing)
│   │   ├── McpContentProcessor.ts (Role: Post-sampling hooks)
│   │   ├── McpServerManager.ts (Class: McpServerManager; Role: Project/User/Dynamic scopes)
│   │   ├── PluginManager.ts (Class: PluginManager; Role: Plugin install/uninstall/toggle)
│   │   └── ToolProcessor.ts (Function: prepareToolForConversation; Role: Schema transformation)
│   ├── plugin
│   │   ├── AgentLoader.ts (Role: Markdown/Frontmatter agent discovery)
│   │   ├── InstalledPluginStore.ts (Class: InstalledPluginStore; Role: V1->V2 database migrations)
│   │   ├── LspManager.ts (Role: Language Server environment expansion)
│   │   └── SkillLoader.ts (Role: .claude/skills directory discovery)
│   ├── sandbox
│   │   └── SandboxSettings.ts (Interface: SandboxConfig; Role: Security policy aggregation)
│   ├── statsig
│   │   ├── NetworkCore.ts (Class: NetworkCore; Role: Retry/Compression for analytics)
│   │   ├── StatsigClientBase.ts (Class: StatsigClient; Features: Gates, Experiments)
│   │   ├── StatsigManager.ts (Functions: checkGate, getExperimentValue)
│   │   ├── StatsigMetadataProvider.ts (Role: Session rotation/duration tracking)
│   │   └── StatsigService.ts (Class: Log; Role: Init markers and diagnostics)
│   ├── teams
│   │   ├── TeamIdentity.ts (Functions: createAgentId, normalizeTeamName)
│   │   ├── TeamManager.ts (Interface: TeamConfig; Role: Team FS structure)
│   │   ├── TeamMessages.ts (Payloads: Join, Shutdown, PlanApproval; Role: Swarm protocol)
│   │   └── TeammateMailbox.ts (Interface: TeammateMailboxMessage; Role: File-based IPC locking)
│   ├── telemetry
│   │   ├── ContextMetrics.ts (Role: Tracking system prompt & git status sizes)
│   │   ├── OtelInit.ts (Role: OpenTelemetry Meter/Tracer bootstrap)
│   │   ├── OtlpConfig.ts (Role: OTLP exporter HTTP/Compression settings)
│   │   ├── SentryHub.ts (Class: SentryHub; Role: Exception breadcrumbs/scope)
│   │   ├── Telemetry.ts (Function: track; Role: Event buffering and flushing)
│   │   ├── Tracing.ts (Class: W3CBaggagePropagator; Role: Context/Baggage injection)
│   │   ├── TracingManager.ts (Functions: startInteractionSpan, endLLMRequestSpan)
│   │   └── TracingUtils.ts (Role: Sentry-compatible baggage headers)
│   ├── terminal
│   │   ├── AppInitializer.ts (Role: Global service orchestration on boot)
│   │   ├── AppleTerminalService.ts (Role: macOS 'defaults' settings backup/restore)
│   │   ├── CommandRegistry.ts (Class: CommandRegistry; Role: Slash command storage)
│   │   ├── ControlStreamClient.ts (Class: ControlStreamClient; Role: WebSocket -> Stream bridge)
│   │   ├── ControlStreamService.ts (Class: ControlStreamService; Logic: JSON-RPC over lines)
│   │   ├── FileIndexService.ts (Function: suggestFiles; Role: git ls-files wrapper)
│   │   ├── HistoryService.ts (Interface: HistoryEntry; Role: SQLite/File history persistence)
│   │   ├── NotificationService.tsx (Provider: NotificationProvider; Role: UI Toast management)
│   │   ├── PermissionService.ts (Function: checkToolPermissions; Role: Logic-based rules)
│   │   ├── ShellSnapshotService.ts (Role: Environment function/alias extraction)
│   │   ├── SwarmBackendRegistry.ts (Role: Tmux vs. iTerm2 backend detection)
│   │   ├── SwarmConstants.ts (Constants: Session/Window names)
│   │   ├── TaskManager.ts (Function: waitForTask; Role: Background process polling)
│   │   ├── TmuxBackend.ts (Class: TmuxBackend; Role: Pane creation, rebalancing, titles)
│   │   ├── WebSocketTransport.ts (Class: WebSocketTransport; Role: Heartbeat/Auto-reconnect)
│   │   └── schemas.ts (Zod Schemas: ToolPermissionResponse, HookCallback)
│   └── tools
│       └── ToolExecutionManager.ts (Class: ToolExecutionManager; Role: Parallel/Serial concurrency)
├── tools
│   ├── EnterPlanModeTool.tsx (Tool: enter_plan_mode; Views: Result/Rejected)
│   ├── FileReadTool
│   │   └── prompt.tsx (Deobfuscated logic: Sandbox settings, Agent FS paths)
│   ├── FileWriteTool
│   │   └── prompt.tsx (Deobfuscated logic: Shell snapshot script generation)
│   ├── TeammateTool.ts (Tool: teammate; Operations: spawnTeam, broadcast, join)
│   ├── ToolGroups.ts (Interface: ToolGroup; Categories: Read, Edit, Execution)
│   └── registry.ts (Role: Centralized JSON Schema repository for tools)
├── types
│   ├── AgentTypes.ts (Types: AgentMode, AgentMessage)
│   └── ambient.d.ts (Ambient: Lexer, Micromatch definitions)
├── utils
│   ├── auth
│   │   └── authHelpers.ts (Function: getAccessTokenData, isFirstParty)
│   ├── browser.ts (Role: CLI wrappers for plugin management commands)
│   ├── cleanup.ts (Placeholder: Terminal exit handlers)
│   ├── config
│   │   └── apiConfig.ts (Role: Base URL and Endpoint constants)
│   ├── fs
│   │   ├── FileParser.ts (Interface: ToolHandler; Parsers: YAML, JSON5, Images)
│   │   ├── Watcher.ts (Functions: initializeFileWatcher, markInternalWrite)
│   │   ├── binaryUtils.ts (Functions: atomicallyInstallBinary, isFileExecutable)
│   │   ├── cache.ts (Functions: processFileRecursively, scanDirectoryForRules)
│   │   ├── copyMoveDelete.ts (Native wrappers: removeSync, outputJson, move)
│   │   ├── fileProcessor.ts (Functions: readFileContent, parseContent)
│   │   ├── pathSanitizer.ts (Functions: sanitizeFilePath, handleBackslashes)
│   │   └── paths.ts (Functions: resolvePathFromHome, getProjectRoot)
│   ├── http
│   │   ├── FetchReadableStream.ts (Class: FetchReadableStream; Role: Node Stream -> Web Body)
│   │   ├── HeaderUtils.ts (Interface: NullableHeaders; Function: normalizeHeaders)
│   │   ├── PathBuilder.ts (Function: createPathBuilder; Role: URI encoding)
│   │   ├── RetryStrategy.ts (Function: requestWithRetry; Logic: Jittered backoff)
│   │   ├── StreamingResponse.ts (Class: StreamingResponse; Role: SSE chunk handling)
│   │   └── nodeHttpShim.ts (Function: urlToHttpOptions; Role: Node native HTTP compatibility)
│   ├── json
│   │   ├── BalancedParser.ts (Function: parseBalancedJSON; Role: LLM-Fix-JSON)
│   │   └── rpc.ts (Functions: isJsonRpcRequest, isJsonRpcResponse)
│   ├── json.ts (Function: parse; Features: Comments and Trailing Comma support)
│   ├── marketplace
│   │   └── SourceParser.ts (Type: MarketplaceSource; Role: GitHub/Git/File path parsing)
│   ├── nativeHostRunner.ts (Role: Chrome Native Messaging protocol implementation)
│   ├── networkUtils.ts (Role: SDK logger configurations)
│   ├── platform
│   │   ├── browserMocks.ts (Role: Globals for window/navigator in Node)
│   │   ├── detector.ts (Functions: getPlatform, isWsl, isMuslEnvironment)
│   │   ├── environment.ts (Functions: isDocker, getTerminalEmulator)
│   │   ├── installer.ts (Function: updateExecutableLink; Role: Symlink maintenance)
│   │   └── shell.ts (Functions: detectShellPath, createShellConfigSnapshot)
│   ├── process
│   │   ├── ProcessLock.ts (Functions: tryAcquireLock, isLockStale; Role: Version concurrency)
│   │   └── commandRunner.ts (Function: runCommand; Interface: CommandResult)
│   ├── releaseUtils.ts (Functions: downloadAndVerify, fetchVersionFromChannel)
│   ├── shared
│   │   ├── bashUtils.ts (Functions: executeBashCommand, cleanSandboxViolations)
│   │   ├── collections.ts (Classes: PriorityQueue, LRUCache)
│   │   ├── commandStringProcessing.ts (Functions: tokenizeCommand, reconstructCommand, isCommandSafe)
│   │   ├── constants.ts (Shared: Signal maps, Exit codes)
│   │   ├── conversationUtils.ts (Functions: updateTokenUsage, adjustMaxTokens)
│   │   ├── crypto.ts (Functions: sha256, sha256Hex)
│   │   ├── displayUtils.ts (Functions: formatDuration, truncate)
│   │   ├── envContext.ts (Functions: getTelemetryContext, normalizeFileExtension)
│   │   ├── envUtils.ts (Functions: initializeSystem, createConnection)
│   │   ├── errors.ts (Errors: RequestAbortedError, NotSupportedError)
│   │   ├── fileHistory.ts (Functions: trackFileModification, createSnapshot; Role: Undo/Rewind)
│   │   ├── fileTypeUtils.ts (Function: determineFileType; Role: Session vs. Memory detection)
│   │   ├── fixtureUtils.ts (Function: withFixtures; Role: API recording/playback)
│   │   ├── formatUri.ts (Functions: formatUri, formatLocation, groupResultsByUri)
│   │   ├── jsonSchema.ts (Class: JSONSchemaGenerator; Role: Zod -> Tool Schema converter)
│   │   ├── lodashUtils.ts (Functions: isObject, isFunction, getValue)
│   │   ├── loggingUtils.ts (Type: SeverityLevel; Role: Log level normalization)
│   │   ├── lspFormatter.ts (Functions: formatLspResult, getSymbolKindString)
│   │   ├── messageUtils.ts (Functions: extractMessageText, normalizeToolResult)
│   │   ├── product.ts (Functions: getProductName, getSystemUser)
│   │   ├── runtime.ts (Classes: ZodString, ZodObject, ZodType; Role: Core validation system)
│   │   ├── runtimeAndEnv.ts (Functions: getBaseConfigDir, getSessionId, getClaudePaths)
│   │   ├── runtimeValidation.ts (Functions: isIsoDate, isPromise)
│   │   ├── rxjsOperators.ts (Classes: Observable, Subscriber, Notification; Role: Internal RxJS)
│   │   ├── sandbox.ts (Function: generateSandboxPolicy; Role: macOS sandbox-exec rules)
│   │   ├── schemaRegistry.ts (Class: Registry; Role: Schema ID lookup)
│   │   ├── screenshotUtils.ts (Functions: getScreenshotFromClipboard, hasPngInClipboard)
│   │   ├── timeUtils.ts (Functions: timestampInSeconds, sleep)
│   │   └── uiUtils.ts (Functions: getTTY, getInteractiveOptions)
│   ├── style.ts (Functions: saveEndpointConfig, validateServerConnection; Role: MCP state)
│   ├── terminal
│   │   ├── Suggestions.ts (Functions: getSlashCommandSuggestions, formatCommandSuggestion)
│   │   ├── iterm2Setup.ts (Functions: findPythonPackageManager, verifyIt2Setup)
│   │   ├── terminalSupport.ts (Function: isTerminalSupported)
│   │   └── tmuxUtils.ts (Functions: hasTmuxSession, ensureTmuxSession)
│   └── text
│       ├── MarkdownRenderer.ts (Class: TerminalMarkdownRenderer; Role: Ink-compatible rendering)
│       ├── ansi.ts (Function: stripAnsi)
│       ├── tokens.ts (Function: truncateString; Role: Visual length truncation)
│       └── width.ts (Function: stringWidth; Role: CJK/Emoji aware width calculation)
└── vendor
    └── Fuse.ts (External Dependency: Fuzzy search implementation)