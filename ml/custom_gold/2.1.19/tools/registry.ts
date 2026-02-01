/**
 * File: src/tools/registry.ts
 * Role: Central registry for all tool schemas and definitions.
 * 
 * Note: We store JSON Schema definitions directly to ensure deterministic
 * sdk-tools.d.ts generation that matches the golden reference.
 */

const bashDescription = [
    "Clear, concise description of what this command does in active voice. Never use words like \"complex\" or \"risk\" in the description - just describe what it does.",
    "",
    "For simple commands (git, npm, standard CLI tools), keep it brief (5-10 words):",
    "- ls → \"List files in current directory\"",
    "- git status → \"Show working tree status\"",
    "- npm install → \"Install package dependencies\"",
    "",
    "For commands that are harder to parse at a glance (piped commands, obscure flags, etc.), add enough context to clarify what it does:",
    "- find . -name \"*.tmp\" -exec rm {} \\; → \"Find and delete all .tmp files recursively\"",
    "- git reset --hard origin/main → \"Discard all local changes and match remote main\"",
    "- curl -s url | jq '.data[]' → \"Fetch JSON from URL and extract data array elements\""
].join("\n");

const questionOptionsDescription = [
    "The available choices for this question. Must have 2-4 options. Each option should be a distinct, mutually exclusive choice (unless multiSelect is enabled). There should be no 'Other' option, that will be provided automatically.",
    "",
    "@minItems 2",
    "@maxItems 4"
].join("\n");

const questionsDescription = [
    "Questions to ask the user (1-4 questions)",
    "",
    "@minItems 1",
    "@maxItems 4"
].join("\n");

const optionSchema = {
    type: "object",
    properties: {
        label: {
            type: "string",
            description: "The display text for this option that the user will see and select. Should be concise (1-5 words) and clearly describe the choice."
        },
        description: {
            type: "string",
            description: "Explanation of what this option means or what will happen if chosen. Useful for providing context about trade-offs or implications."
        }
    },
    required: ["label", "description"],
    additionalProperties: false
};

const optionsTuple2 = {
    type: "array",
    items: [optionSchema, optionSchema],
    minItems: 2,
    maxItems: 2,
    additionalItems: false
};

const optionsTuple3 = {
    type: "array",
    items: [optionSchema, optionSchema, optionSchema],
    minItems: 3,
    maxItems: 3,
    additionalItems: false
};

const optionsTuple4 = {
    type: "array",
    items: [optionSchema, optionSchema, optionSchema, optionSchema],
    minItems: 4,
    maxItems: 4,
    additionalItems: false
};

const optionsSchema = {
    description: questionOptionsDescription,
    anyOf: [optionsTuple2, optionsTuple3, optionsTuple4]
};

const questionSchema = {
    type: "object",
    properties: {
        question: {
            type: "string",
            description: "The complete question to ask the user. Should be clear, specific, and end with a question mark. Example: \"Which library should we use for date formatting?\" If multiSelect is true, phrase it accordingly, e.g. \"Which features do you want to enable?\""
        },
        header: {
            type: "string",
            description: "Very short label displayed as a chip/tag (max 12 chars). Examples: \"Auth method\", \"Library\", \"Approach\"."
        },
        options: optionsSchema,
        multiSelect: {
            type: "boolean",
            description: "Set to true to allow the user to select multiple options instead of just one. Use when choices are not mutually exclusive."
        }
    },
    required: ["question", "header", "options", "multiSelect"],
    additionalProperties: false
};

const questionsTuple1 = {
    type: "array",
    items: [questionSchema],
    minItems: 1,
    maxItems: 1,
    additionalItems: false
};

const questionsTuple2 = {
    type: "array",
    items: [questionSchema, questionSchema],
    minItems: 2,
    maxItems: 2,
    additionalItems: false
};

const questionsTuple3 = {
    type: "array",
    items: [questionSchema, questionSchema, questionSchema],
    minItems: 3,
    maxItems: 3,
    additionalItems: false
};

const questionsTuple4 = {
    type: "array",
    items: [questionSchema, questionSchema, questionSchema, questionSchema],
    minItems: 4,
    maxItems: 4,
    additionalItems: false
};

const questionsSchema = {
    description: questionsDescription,
    anyOf: [questionsTuple1, questionsTuple2, questionsTuple3, questionsTuple4]
};

/**
 * Combined schemas for all tools. Used by generate-types.ts.
 */
export const AllToolSchemas: Record<string, any> = {
    Agent: {
        type: "object",
        properties: {
            description: {
                type: "string",
                description: "A short (3-5 word) description of the task"
            },
            prompt: {
                type: "string",
                description: "The task for the agent to perform"
            },
            subagent_type: {
                type: "string",
                description: "The type of specialized agent to use for this task"
            },
            model: {
                type: "string",
                enum: ["sonnet", "opus", "haiku"],
                description: "Optional model to use for this agent. If not specified, inherits from parent. Prefer haiku for quick, straightforward tasks to minimize cost and latency."
            },
            resume: {
                type: "string",
                description: "Optional agent ID to resume from. If provided, the agent will continue from the previous execution transcript."
            },
            run_in_background: {
                type: "boolean",
                description: "Set to true to run this agent in the background. The tool result will include an output_file path - use Read tool or Bash tail to check on output."
            },
            max_turns: {
                type: "number",
                description: "Maximum number of agentic turns (API round-trips) before stopping. Used internally for warmup."
            },
            allowed_tools: {
                type: "array",
                items: {
                    type: "string"
                },
                description: "Tools to grant this agent. User will be prompted to approve if not already allowed. Example: [\"Bash(git commit*)\", \"Read\"]"
            },
            name: {
                type: "string",
                description: "Name for the spawned agent"
            },
            team_name: {
                type: "string",
                description: "Team name for spawning. Uses current team context if omitted."
            },
            mode: {
                type: "string",
                enum: ["acceptEdits", "bypassPermissions", "default", "delegate", "dontAsk", "plan"],
                description: "Permission mode for spawned teammate (e.g., \"plan\" to require plan approval)."
            }
        },
        required: ["description", "prompt", "subagent_type"],
        additionalProperties: false
    },
    Bash: {
        type: "object",
        properties: {
            command: {
                type: "string",
                description: "The command to execute"
            },
            timeout: {
                type: "number",
                description: "Optional timeout in milliseconds (max 600000)"
            },
            description: {
                type: "string",
                description: bashDescription
            },
            run_in_background: {
                type: "boolean",
                description: "Set to true to run this command in the background. Use TaskOutput to read the output later."
            },
            dangerouslyDisableSandbox: {
                type: "boolean",
                description: "Set this to true to dangerously override sandbox mode and run commands without sandboxing."
            },
            _simulatedSedEdit: {
                type: "object",
                description: "Internal: pre-computed sed edit result from preview",
                properties: {
                    filePath: {
                        type: "string"
                    },
                    newContent: {
                        type: "string"
                    }
                },
                required: ["filePath", "newContent"],
                additionalProperties: false
            }
        },
        required: ["command"],
        additionalProperties: false
    },
    TaskOutput: {
        type: "object",
        properties: {
            task_id: {
                type: "string",
                description: "The task ID to get output from"
            },
            block: {
                type: "boolean",
                description: "Whether to wait for completion"
            },
            timeout: {
                type: "number",
                description: "Max wait time in ms"
            }
        },
        required: ["task_id", "block", "timeout"],
        additionalProperties: false
    },
    ExitPlanMode: {
        type: "object",
        properties: {
            allowedPrompts: {
                type: "array",
                description: "Prompt-based permissions needed to implement the plan. These describe categories of actions rather than specific commands.",
                items: {
                    type: "object",
                    properties: {
                        tool: {
                            type: "string",
                            enum: ["Bash"],
                            description: "The tool this prompt applies to"
                        },
                        prompt: {
                            type: "string",
                            description: "Semantic description of the action, e.g. \"run tests\", \"install dependencies\""
                        }
                    },
                    required: ["tool", "prompt"],
                    additionalProperties: false
                }
            },
            pushToRemote: {
                type: "boolean",
                description: "Whether to push the plan to a remote Claude.ai session"
            },
            remoteSessionId: {
                type: "string",
                description: "The remote session ID if pushed to remote"
            },
            remoteSessionUrl: {
                type: "string",
                description: "The remote session URL if pushed to remote"
            },
            remoteSessionTitle: {
                type: "string",
                description: "The remote session title if pushed to remote"
            },
            launchSwarm: {
                type: "boolean",
                description: "Whether to launch a swarm to implement the plan"
            },
            teammateCount: {
                type: "number",
                description: "Number of teammates to spawn in the swarm"
            }
        },
        required: [],
        additionalProperties: true
    },
    FileEdit: {
        type: "object",
        properties: {
            file_path: {
                type: "string",
                description: "The absolute path to the file to modify"
            },
            old_string: {
                type: "string",
                description: "The text to replace"
            },
            new_string: {
                type: "string",
                description: "The text to replace it with (must be different from old_string)"
            },
            replace_all: {
                type: "boolean",
                description: "Replace all occurences of old_string (default false)"
            }
        },
        required: ["file_path", "old_string", "new_string"],
        additionalProperties: false
    },
    FileRead: {
        type: "object",
        properties: {
            file_path: {
                type: "string",
                description: "The absolute path to the file to read"
            },
            offset: {
                type: "number",
                description: "The line number to start reading from. Only provide if the file is too large to read at once"
            },
            limit: {
                type: "number",
                description: "The number of lines to read. Only provide if the file is too large to read at once."
            }
        },
        required: ["file_path"],
        additionalProperties: false
    },
    FileWrite: {
        type: "object",
        properties: {
            file_path: {
                type: "string",
                description: "The absolute path to the file to write (must be absolute, not relative)"
            },
            content: {
                type: "string",
                description: "The content to write to the file"
            }
        },
        required: ["file_path", "content"],
        additionalProperties: false
    },
    Glob: {
        type: "object",
        properties: {
            pattern: {
                type: "string",
                description: "The glob pattern to match files against"
            },
            path: {
                type: "string",
                description: "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided."
            }
        },
        required: ["pattern"],
        additionalProperties: false
    },
    Grep: {
        type: "object",
        properties: {
            pattern: {
                type: "string",
                description: "The regular expression pattern to search for in file contents"
            },
            path: {
                type: "string",
                description: "File or directory to search in (rg PATH). Defaults to current working directory."
            },
            glob: {
                type: "string",
                description: "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"
            },
            output_mode: {
                type: "string",
                enum: ["content", "files_with_matches", "count"],
                description: "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."
            },
            "-B": {
                type: "number",
                description: "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."
            },
            "-A": {
                type: "number",
                description: "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."
            },
            "-C": {
                type: "number",
                description: "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."
            },
            "-n": {
                type: "boolean",
                description: "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true."
            },
            "-i": {
                type: "boolean",
                description: "Case insensitive search (rg -i)"
            },
            type: {
                type: "string",
                description: "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."
            },
            head_limit: {
                type: "number",
                description: "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 0 (unlimited)."
            },
            offset: {
                type: "number",
                description: "Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Works across all output modes. Defaults to 0."
            },
            multiline: {
                type: "boolean",
                description: "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."
            }
        },
        required: ["pattern"],
        additionalProperties: false
    },
    TaskStop: {
        type: "object",
        properties: {
            task_id: {
                type: "string",
                description: "The ID of the background task to stop"
            },
            shell_id: {
                type: "string",
                description: "Deprecated: use task_id instead"
            }
        },
        required: [],
        additionalProperties: false
    },
    ListMcpResources: {
        type: "object",
        properties: {
            server: {
                type: "string",
                description: "Optional server name to filter resources by"
            }
        },
        required: [],
        additionalProperties: false
    },
    Mcp: {
        type: "object",
        additionalProperties: true
    },
    NotebookEdit: {
        type: "object",
        properties: {
            notebook_path: {
                type: "string",
                description: "The absolute path to the Jupyter notebook file to edit (must be absolute, not relative)"
            },
            cell_id: {
                type: "string",
                description: "The ID of the cell to edit. When inserting a new cell, the new cell will be inserted after the cell with this ID, or at the beginning if not specified."
            },
            new_source: {
                type: "string",
                description: "The new source for the cell"
            },
            cell_type: {
                type: "string",
                enum: ["code", "markdown"],
                description: "The type of the cell (code or markdown). If not specified, it defaults to the current cell type. If using edit_mode=insert, this is required."
            },
            edit_mode: {
                type: "string",
                enum: ["replace", "insert", "delete"],
                description: "The type of edit to make (replace, insert, delete). Defaults to replace."
            }
        },
        required: ["notebook_path", "new_source"],
        additionalProperties: false
    },
    ReadMcpResource: {
        type: "object",
        properties: {
            server: {
                type: "string",
                description: "The MCP server name"
            },
            uri: {
                type: "string",
                description: "The resource URI to read"
            }
        },
        required: ["server", "uri"],
        additionalProperties: false
    },
    TodoWrite: {
        type: "object",
        properties: {
            todos: {
                type: "array",
                description: "The updated todo list",
                items: {
                    type: "object",
                    properties: {
                        content: {
                            type: "string"
                        },
                        status: {
                            type: "string",
                            enum: ["pending", "in_progress", "completed"]
                        },
                        activeForm: {
                            type: "string"
                        }
                    },
                    required: ["content", "status", "activeForm"],
                    additionalProperties: false
                }
            }
        },
        required: ["todos"],
        additionalProperties: false
    },
    WebFetch: {
        type: "object",
        properties: {
            url: {
                type: "string",
                description: "The URL to fetch content from"
            },
            prompt: {
                type: "string",
                description: "The prompt to run on the fetched content"
            }
        },
        required: ["url", "prompt"],
        additionalProperties: false
    },
    WebSearch: {
        type: "object",
        properties: {
            query: {
                type: "string",
                description: "The search query to use"
            },
            allowed_domains: {
                type: "array",
                items: {
                    type: "string"
                },
                description: "Only include search results from these domains"
            },
            blocked_domains: {
                type: "array",
                items: {
                    type: "string"
                },
                description: "Never include search results from these domains"
            }
        },
        required: ["query"],
        additionalProperties: false
    },
    AskUserQuestion: {
        type: "object",
        properties: {
            questions: questionsSchema,
            answers: {
                type: "object",
                description: "User answers collected by the permission component",
                additionalProperties: {
                    type: "string"
                }
            },
            metadata: {
                type: "object",
                description: "Optional metadata for tracking and analytics purposes. Not displayed to user.",
                properties: {
                    source: {
                        type: "string",
                        description: "Optional identifier for the source of this question (e.g., \"remember\" for /remember command). Used for analytics tracking."
                    }
                },
                required: [],
                additionalProperties: false
            }
        },
        required: ["questions"],
        additionalProperties: false
    },
    Config: {
        type: "object",
        properties: {
            setting: {
                type: "string",
                description: "The setting key (e.g., \"theme\", \"model\", \"permissions.defaultMode\")"
            },
            value: {
                description: "The new value. Omit to get current value.",
                anyOf: [
                    {
                        type: "string"
                    },
                    {
                        type: "boolean"
                    },
                    {
                        type: "number"
                    }
                ]
            }
        },
        required: ["setting"],
        additionalProperties: false
    },
    TaskCreate: {
        type: "object",
        properties: {
            subject: {
                type: "string",
                description: "The title/subject of the task"
            },
            description: {
                type: "string",
                description: "Detailed description of the task"
            }
        },
        required: ["subject"],
        additionalProperties: false
    },
    TaskUpdate: {
        type: "object",
        properties: {
            id: {
                type: "string",
                description: "The ID of the task to update"
            },
            status: {
                type: "string",
                enum: ["pending", "in_progress", "completed", "cancelled"],
                description: "New status"
            },
            subject: {
                type: "string",
                description: "New subject"
            },
            description: {
                type: "string",
                description: "New description"
            }
        },
        required: ["id"],
        additionalProperties: false
    },
    TaskList: {
        type: "object",
        properties: {
            status: {
                type: "string",
                enum: ["pending", "in_progress", "completed", "cancelled", "all"],
                description: "Filter by status (default: all)"
            }
        },
        required: [],
        additionalProperties: false
    },
    MCPSearch: {
        type: "object",
        properties: {
            query: {
                type: "string",
                description: "Search query for MCP tools/resources"
            }
        },
        required: ["query"],
        additionalProperties: false
    }

};
