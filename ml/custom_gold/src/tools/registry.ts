import { BashTool } from './bash/BashTool.js';
import { FileReadTool } from './definitions/fileRead.js';
import { FileWriteTool } from './definitions/fileWrite.js';
import { FileEditTool } from './definitions/fileEdit.js';
import { GrepSearchTool, FileSearchTool } from './definitions/search.js';
import { TaskOutputTool } from './definitions/taskOutput.js';
import { KillShellTool } from './definitions/killShell.js';
import { AskUserQuestionTool } from './terminal/AskUserQuestionTool.js';
import { NotebookEditTool } from './definitions/NotebookEditTool.js';
import { AgentTool } from './definitions/agent.js';
import { McpInputSchema } from './definitions/mcp.js';
import { WebSearchTool } from './web/WebSearchTool.js';
import { ExitPlanModeTool } from './terminal/ExitPlanModeTool.js';
import { FetchTool } from './definitions/FetchTool.js';
import { todoWriteTool } from './definitions/todoWrite.js';
import { ListMcpResourcesTool, ReadMcpResourceTool } from './mcp/McpResourceTools.js';

export const AllToolSchemas = {
    Agent: AgentTool.inputSchema,
    Bash: BashTool.inputSchema,
    TaskOutput: TaskOutputTool.inputSchema,
    ExitPlanMode: ExitPlanModeTool.inputSchema,
    FileEdit: FileEditTool.inputSchema,
    FileRead: FileReadTool.inputSchema,
    FileWrite: FileWriteTool.inputSchema,
    Glob: FileSearchTool.inputSchema,
    Grep: GrepSearchTool.inputSchema,
    KillShell: KillShellTool.inputSchema,
    ListMcpResources: ListMcpResourcesTool.inputSchema,
    Mcp: McpInputSchema,
    NotebookEdit: NotebookEditTool.inputSchema,
    ReadMcpResource: ReadMcpResourceTool.inputSchema,
    TodoWrite: todoWriteTool.inputSchema,
    WebFetch: FetchTool.inputSchema,
    WebSearch: WebSearchTool.inputSchema,
    AskUserQuestion: AskUserQuestionTool.inputSchema
};
