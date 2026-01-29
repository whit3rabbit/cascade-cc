/**
 * File: src/commands/SlashCommandDispatcher.ts
 * Role: Central dispatcher for handling slash commands in the REPL.
 */

/**
 * File: src/commands/SlashCommandDispatcher.ts
 * Role: Central dispatcher for handling slash commands in the REPL.
 */

export interface CommandContext {
    setMessages: (updater: (prev: any[]) => any[]) => void;
    setPlanMode: (updater: (prev: boolean) => boolean) => void;
    setVimModeEnabled: (updater: (prev: boolean) => boolean) => void;
    setShowTasks: (updater: (prev: boolean) => boolean) => void;
    setIsTyping: (isTyping: boolean) => void;
    exit: () => void;
    cwd: string;
}

export class SlashCommandDispatcher {
    static async handleCommand(input: string, context: CommandContext): Promise<boolean> {
        const cmd = input.trim().split(' ')[0];
        const args = input.trim().split(' ').slice(1).join(' ');

        switch (cmd) {
            case '/clear':
                context.setMessages(() => []);
                return true;

            case '/exit':
                context.exit();
                return true;

            case '/tasks':
            case '/todos':
                context.setShowTasks(prev => !prev);
                return true;

            case '/plan':
                context.setPlanMode(prev => !prev);
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Plan mode toggled.' }]);
                return true;

            case '/vim':
                context.setVimModeEnabled(prev => !prev);
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Vim mode toggled.' }]);
                return true;

            case '/help':
                context.setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `Available commands:
- /clear: Clear conversation history
- /compact: Compact conversation history (Stub)
- /config: Open configuration (Stub)
- /cost: Show token usage (Stub)
- /doctor: Health check
- /exit: Exit the session
- /help: Show this help message
- /init: Initialize project
- /mcp: Manage MCP servers (Stub)
- /plan: Toggle Planning Mode
- /tasks: Toggle task list
- /vim: Toggle Vim mode
`
                }]);
                return true;

            case '/init':
                // For now, we'll let this fall through to the LLM if we want actual behavior, 
                // or just stub it here. The prompt-based init is better handled by content generation.
                // context.setMessages(prev => [...prev, { role: 'assistant', content: 'Initialization logic would run here. (Falling through to LLM for now)' }]);
                // return false; 

                // MOCK Implementation
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Checking project structure... (Mock Init)' }]);
                return true;

            case '/doctor':
                context.setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `**Claude Code Doctor**
- Version: 2.1.19
- Node: ${process.version}
- Platform: ${process.platform}
- CWD: ${context.cwd}
- Connectivity: OK (Mock)
- Auth: OK (Mock)`
                }]);
                return true;

            case '/compact':
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Compacting conversation history... (Done)' }]);
                return true;

            case '/context':
                // In a real app, this would show a visual breakdown using Ink
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Context usage: 15% (Mock)' }]);
                return true;

            default:
                // Return false to indicate not handled (passed to LLM if not a command, but here we assume all / are commands)
                // Actually, if it starts with /, we should probably say "Unknown command" if not found, 
                // UNLESS it's meant to be an MCP command passed to the LLM. 
                // For now, let's return false and let the caller decide or show error.
                if (cmd.startsWith('/')) {
                    context.setMessages(prev => [...prev, { role: 'assistant', content: `Unknown command: ${cmd}` }]);
                    return true;
                }
                return false;
        }
    }
}
