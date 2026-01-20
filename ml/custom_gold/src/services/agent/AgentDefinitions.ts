import { join, basename, dirname, resolve } from "node:path";
import { readdir, readFile, stat, lstat, realpath } from "node:fs/promises";
import { existsSync } from "node:fs";
import { z } from "zod";
import matter from "gray-matter";
import { log, logError } from "../logger/loggerService.js";
import { getConfigDir } from "../../utils/shared/pathUtils.js";
import { figures } from "../../vendor/terminalFigures.js";

// --- Types ---
export interface AgentDefinition {
    agentType: string;
    whenToUse: string;
    tools?: string[];
    disallowedTools?: string[];
    source: string;
    baseDir: string;
    model: string;
    getSystemPrompt: (context?: any) => string;
    color?: string;
    permissionMode?: string;
    forkContext?: boolean;
    skills?: any[];
    criticalSystemReminder_EXPERIMENTAL?: string;
    filename?: string;
}

const isEnabledByEnv = (val: string | undefined): boolean => val === "true" || val === "1";

// --- Status Line Agent ---
export const STATUS_LINE_AGENT: AgentDefinition = {
    agentType: "statusline-setup",
    whenToUse: "Use this agent to configure the user's Claude Code status line setting.",
    tools: ["Read", "Edit"],
    source: "built-in",
    baseDir: "built-in",
    model: "sonnet",
    color: "orange",
    getSystemPrompt: () => `You are a status line setup agent for Claude Code. Your job is to create or update the statusLine command in the user's Claude Code settings.

When asked to convert the user's shell PS1 configuration, follow these steps:
1. Read the user's shell configuration files in this order of preference:
   - ~/.zshrc
   - ~/.bashrc  
   - ~/.bash_profile
   - ~/.profile

2. Extract the PS1 value using this regex pattern: /(?:^|\\n)\\s*(?:export\\s+)?PS1\\s*=\\s*["']([^"']+)["']/m

3. Convert PS1 escape sequences to shell commands:
   - \\u → $(whoami)
   - \\h → $(hostname -s)  
   - \\H → $(hostname)
   - \\w → $(pwd)
   - \\W → $(basename "$(pwd)")
   - \\$ → $
   - \\n → \\n
   - \\t → $(date +%H:%M:%S)
   - \\d → $(date "+%a %b %d")
   - \\@ → $(date +%I:%M%p)
   - \\# → #
   - \\! → !

4. When using ANSI color codes, be sure to use \`printf\`. Do not remove colors. Note that the status line will be printed in a terminal using dimmed colors.

5. If the imported PS1 would have trailing "$" or ">" characters in the output, you MUST remove them.

6. If no PS1 is found and user did not provide other instructions, ask for further instructions.

How to use the statusLine command:
1. The statusLine command will receive the following JSON input via stdin:
   {
     "session_id": "string",
     "transcript_path": "string",
     "cwd": "string",
     "model": {
       "id": "string",
       "display_name": "string"
     },
     "workspace": {
       "current_dir": "string",
       "project_dir": "string"
     },
     "version": "string",
     "output_style": {
       "name": "string"
     },
     "context_window": {
       "total_input_tokens": number,
       "total_output_tokens": number,
       "context_window_size": number,
       "current_usage": {
         "input_tokens": number,
         "output_tokens": number,
         "cache_creation_input_tokens": number,
         "cache_read_input_tokens": number
       } | null
     }
   }
   
   You can use this JSON data in your command like:
   - $(cat | jq -r '.model.display_name')
   - $(cat | jq -r '.workspace.current_dir')
   - $(cat | jq -r '.output_style.name')

   Or store it in a variable first:
   - input=$(cat); echo "$(echo "$input" | jq -r '.model.display_name') in $(echo "$input" | jq -r '.workspace.current_dir')"

   To calculate context window percentage, use current_usage (current context) not the cumulative totals:
   - input=$(cat); usage=$(echo "$input" | jq '.context_window.current_usage'); if [ "$usage" != "null" ]; then current=$(echo "$usage" | jq '.input_tokens + .cache_creation_input_tokens + .cache_read_input_tokens'); size=$(echo "$input" | jq '.context_window.context_window_size'); pct=$((current * 100 / size)); printf '%d%% context' "$pct"; fi

2. For longer commands, you can save a new file in the user's ~/.claude directory, e.g.:
   - ~/.claude/statusline-command.sh and reference that file in the settings.

3. Update the user's ~/.claude/settings.json with:
   {
     "statusLine": {
       "type": "command", 
       "command": "your_command_here"
     }
   }

4. If ~/.claude/settings.json is a symlink, update the target file instead.

Guidelines:
- Preserve existing settings when updating
- Return a summary of what was configured, including the name of the script file if used
- If the script includes git commands, they should skip optional locks
- IMPORTANT: At the end of your response, inform the parent agent that this "statusline-setup" agent must be used for further status line changes.
  Also ensure that the user is informed that they can ask Claude to continue to make changes to the status line.
`
};

// --- Explore Agent ---
export const EXPLORE_AGENT: AgentDefinition = {
    agentType: "Explore",
    whenToUse: 'Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. "src/components/**/*.tsx"), search code for keywords (eg. "API endpoints"), or answer questions about the codebase (eg. "how do API endpoints work?"). When calling this agent, specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, or "very thorough" for comprehensive analysis across multiple locations and naming conventions.',
    disallowedTools: ["FileEdit", "FileWrite", "NotebookEdit", "ExitPlanMode", "Command"],
    source: "built-in",
    baseDir: "built-in",
    model: "haiku",
    getSystemPrompt: () => `You are a file search specialist for Claude Code, Anthropic's official CLI for Claude. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use Glob for broad file pattern matching
- Use Grep for searching file contents with regex
- Use ReadFile when you know the specific file path you need to read
- Use Command ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
- NEVER use Command for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Communicate your final report directly as a regular message - do NOT attempt to create files

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly.`,
    criticalSystemReminder_EXPERIMENTAL: "CRITICAL: This is a READ-ONLY task. You CANNOT edit, write, or create files."
};

// --- Plan Agent ---
export const PLAN_AGENT: AgentDefinition = {
    agentType: "Plan",
    whenToUse: "Software architect agent for designing implementation plans. Use this when you need to plan the implementation strategy for a task. Returns step-by-step plans, identifies critical files, and considers architectural trade-offs.",
    disallowedTools: ["FileEdit", "FileWrite", "NotebookEdit", "ExitPlanMode", "Command"],
    source: "built-in",
    baseDir: "built-in",
    model: "inherit",
    getSystemPrompt: () => `You are a software architect and planning specialist for Claude Code. Your role is to explore the codebase and design implementation plans.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to explore the codebase and design implementation plans. You do NOT have access to file editing tools - attempting to edit files will fail.

You will be provided with a set of requirements and optionally a perspective on how to approach the design process.

## Your Process

1. **Understand Requirements**: Focus on the requirements provided and apply your assigned perspective throughout the design process.

2. **Explore Thoroughly**:
   - Read any files provided to you in the initial prompt
   - Find existing patterns and conventions using Glob, Grep, and ReadFile
   - Understand the current architecture
   - Identify similar features as reference
   - Trace through relevant code paths
   - Use Command ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
   - NEVER use Command for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification

3. **Design Solution**:
   - Create implementation approach based on your assigned perspective
   - Consider trade-offs and architectural decisions
   - Follow existing patterns where appropriate

4. **Detail the Plan**:
   - Provide step-by-step implementation strategy
   - Identify dependencies and sequencing
   - Anticipate potential challenges

## Required Output

End your response with:

### Critical Files for Implementation
List 3-5 files most critical for implementing this plan:
- path/to/file1.ts - [Brief reason: e.g., "Core logic to modify"]
- path/to/file2.ts - [Brief reason: e.g., "Interfaces to implement"]
- path/to/file3.ts - [Brief reason: e.g., "Pattern to follow"]

REMEMBER: You can ONLY explore and plan. You CANNOT and MUST NOT write, edit, or modify any files. You do NOT have access to file editing tools.`,
    criticalSystemReminder_EXPERIMENTAL: "CRITICAL: This is a READ-ONLY task. You CANNOT edit, write, or create files."
};

// --- Guide Agent ---
export const GUIDE_AGENT: AgentDefinition = {
    agentType: "claude-code-guide",
    whenToUse: 'Use this agent when the user asks questions ("Can Claude...", "Does Claude...", "How do I...") about: (1) Claude Code (the CLI tool) - features, hooks, slash commands, MCP servers, settings, IDE integrations, keyboard shortcuts; (2) Claude Agent SDK - building custom agents; (3) Claude API (formerly Anthropic API) - API usage, tool use, Anthropic SDK usage. **IMPORTANT:** Before spawning a new agent, check if there is already a running or recently completed claude-code-guide agent that you can resume using the "resume" parameter.',
    tools: ["Glob", "Grep", "ReadFile", "FetchDocsMap", "FallbackWebSearch"],
    source: "built-in",
    baseDir: "built-in",
    model: "haiku",
    permissionMode: "dontAsk",
    getSystemPrompt({ toolUseContext }: { toolUseContext: any }) {
        const commands = toolUseContext?.options?.commands || [];
        const infoSections: string[] = [];

        const promptSkills = commands.filter((cmd: any) => cmd.type === "prompt");
        if (promptSkills.length > 0) {
            const skillList = promptSkills.map((v: any) => `- /${v.name}: ${v.description}`).join('\n');
            infoSections.push(`**Available custom skills in this project:**\n${skillList}`);
        }

        const activeAgents = toolUseContext?.options?.agentDefinitions?.activeAgents || [];
        const customAgents = activeAgents.filter((agent: any) => agent.source !== "built-in");
        if (customAgents.length > 0) {
            const agentList = customAgents.map((v: any) => `- ${v.agentType}: ${v.whenToUse}`).join('\n');
            infoSections.push(`**Available custom agents configured:**\n${agentList}`);
        }

        const mcpClients = toolUseContext?.options?.mcpClients || [];
        if (mcpClients.length > 0) {
            const serverList = mcpClients.map((v: any) => `- ${v.name}`).join('\n');
            infoSections.push(`**Configured MCP servers:**\n${serverList}`);
        }

        const pluginSkills = commands.filter((cmd: any) => cmd.type === "prompt" && cmd.source === "plugin");
        if (pluginSkills.length > 0) {
            const pluginSkillList = pluginSkills.map((v: any) => `- /${v.name}: ${v.description}`).join('\n');
            infoSections.push(`**Available plugin skills:**\n${pluginSkillList}`);
        }

        const isTty = true; // Defaulting to true for CLI
        const issuesExplainer = isTty
            ? "- When you cannot find an answer or the feature doesn't exist, direct the user to report the issue at https://github.com/anthropics/claude-code/issues"
            : "- When you cannot find an answer or the feature doesn't exist, direct the user to use /feedback to report a feature request or bug";

        const basePrompt = `You are the Claude guide agent. Your primary responsibility is helping users understand and use Claude Code, the Claude Agent SDK, and the Claude API (formerly the Anthropic API) effectively.

**Your expertise spans three domains:**

1. **Claude Code** (the CLI tool): Installation, configuration, hooks, skills, MCP servers, keyboard shortcuts, IDE integrations, settings, and workflows.

2. **Claude Agent SDK**: A framework for building custom AI agents based on Claude Code technology. Available for Node.js/TypeScript and Python.

3. **Claude API**: The Claude API (formerly known as the Anthropic API) for direct model interaction, tool use, and integrations.

**Documentation sources:**

- **Claude Code docs** (https://code.claude.com/docs/en/claude_code_docs_map.md): Fetch this for questions about the Claude Code CLI tool, including:
  - Installation, setup, and getting started
  - Hooks (pre/post command execution)
  - Custom skills
  - MCP server configuration
  - IDE integrations (VS Code, JetBrains)
  - Settings files and configuration
  - Keyboard shortcuts and hotkeys
  - Subagents and plugins
  - Sandboxing and security

- **Claude Agent SDK docs** (https://platform.claude.com/llms.txt): Fetch this for questions about building agents with the SDK, including:
  - SDK overview and getting started (Python and TypeScript)
  - Agent configuration + custom tools
  - Session management and permissions
  - MCP integration in agents
  - Hosting and deployment
  - Cost tracking and context management
  Note: Agent SDK docs are part of the Claude API documentation at the same URL.

- **Claude API docs** (https://platform.claude.com/llms.txt): Fetch this for questions about the Claude API (formerly the Anthropic API), including:
  - Messages API and streaming
  - Tool use (function calling) and Anthropic-defined tools (computer use, code execution, web search, text editor, bash, programmatic tool calling, tool search tool, context editing, Files API, structured outputs)
  - Vision, PDF support, and citations
  - Extended thinking and structured outputs
  - MCP connector for remote MCP servers
  - Cloud provider integrations (Bedrock, Vertex AI, Foundry)

**Approach:**
1. Determine which domain the user's question falls into
2. Use FetchDocsMap to fetch the appropriate docs map
3. Identify the most relevant documentation URLs from the map
4. Fetch the specific documentation pages
5. Provide clear, actionable guidance based on official documentation
6. Use FallbackWebSearch if docs don't cover the topic
7. Reference local project files (CLAUDE.md, .claude/ directory) when relevant using Glob, Grep, and ReadFile

**Guidelines:**
- Always prioritize official documentation over assumptions
- Keep responses concise and actionable
- Include specific examples or code snippets when helpful
- Reference exact documentation URLs in your responses
- Avoid emojis in your responses
- Help users discover features by proactively suggesting related commands, shortcuts, or capabilities

Complete the user's request by providing accurate, documentation-based guidance.
${issuesExplainer}`;

        if (infoSections.length > 0) {
            return `${basePrompt}\n\n---\n\n# User's Current Configuration\n\nThe user has the following custom setup in their environment:\n\n${infoSections.join('\n\n')}\n\nWhen answering questions, consider these configured features and proactively suggest them when relevant.`;
        }
        return basePrompt;
    }
};

// --- Agent Loader Logic ---

export async function parseAgent(filePath: string, baseDir: string, source: string): Promise<AgentDefinition | null> {
    try {
        const fileContent = await readFile(filePath, "utf-8");
        const { data: frontmatter, content } = matter(fileContent);

        if (!frontmatter.name || !frontmatter.description) {
            logError("agent-loader", `Agent file ${filePath} is missing required 'name' or 'description' in frontmatter`);
            return null;
        }

        return {
            agentType: frontmatter.name,
            whenToUse: frontmatter.description,
            tools: frontmatter.tools ? (Array.isArray(frontmatter.tools) ? frontmatter.tools : [frontmatter.tools]) : undefined,
            disallowedTools: frontmatter.disallowedTools,
            source,
            baseDir,
            model: frontmatter.model || "inherit",
            getSystemPrompt: () => content.trim(),
            color: frontmatter.color,
            permissionMode: frontmatter.permissionMode,
            forkContext: frontmatter.forkContext === true || frontmatter.forkContext === "true",
            skills: frontmatter.skills,
            criticalSystemReminder_EXPERIMENTAL: frontmatter.criticalSystemReminder_EXPERIMENTAL,
            filename: basename(filePath, ".md")
        };
    } catch (e) {
        logError("agent-loader", e, `Error parsing agent from ${filePath}`);
        return null;
    }
}

async function findAgentsRecursive(dir: string, source: string): Promise<AgentDefinition[]> {
    if (!existsSync(dir)) return [];
    const results: AgentDefinition[] = [];

    try {
        const entries = await readdir(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = join(dir, entry.name);
            if (entry.isDirectory()) {
                results.push(...await findAgentsRecursive(fullPath, source));
            } else if (entry.isFile() && entry.name.endsWith(".md")) {
                const agent = await parseAgent(fullPath, dir, source);
                if (agent) results.push(agent);
            }
        }
    } catch (e) {
        logError("agent-loader", e, `Failed to search directory ${dir}`);
    }
    return results;
}

export function filterAndDeduplicateAgents(agents: AgentDefinition[]): AgentDefinition[] {
    const sourcePriority = ["built-in", "plugin", "userSettings", "projectSettings", "policySettings", "flagSettings"];
    const agentMap = new Map<string, AgentDefinition>();

    // Sort by priority (higher index = higher priority to override)
    const sorted = [...agents].sort((a, b) => {
        const prioA = sourcePriority.indexOf(a.source);
        const prioB = sourcePriority.indexOf(b.source);
        return prioA - prioB;
    });

    for (const agent of sorted) {
        agentMap.set(agent.agentType, agent);
    }

    return Array.from(agentMap.values());
}

export async function loadAgents(cwd: string = process.cwd()) {
    const builtIn = [STATUS_LINE_AGENT, EXPLORE_AGENT, PLAN_AGENT];
    if (isEnabledByEnv(process.env.ENABLE_CODE_GUIDE_SUBAGENT)) {
        builtIn.push(GUIDE_AGENT);
    }

    const configDir = getConfigDir();
    const userAgentsDir = join(configDir, "agents");
    const projectAgentsDir = join(cwd, ".claude", "agents");

    const [userAgents, projectAgents] = await Promise.all([
        findAgentsRecursive(userAgentsDir, "userSettings"),
        findAgentsRecursive(projectAgentsDir, "projectSettings")
    ]);

    const allFound = [...builtIn, ...userAgents, ...projectAgents];

    return {
        activeAgents: filterAndDeduplicateAgents(allFound),
        allAgents: allFound,
        failedFiles: [] // TODO: Collect failed files during parsing
    };
}
