 
/** Generated file - DO NOT EDIT */

/**
 * JSON Schema definitions for Claude CLI tool inputs
 */
export type ToolInputSchemas =
  | AgentInput
  | BashInput
  | FileEditInput
  | FileReadInput
  | FileWriteInput
  | GrepInput
  | FileSearchInput
  | TaskOutputInput
  | KillShellInput
  | AskUserQuestionInput
  | NotebookEditInput
  | McpInput
  | WebSearchInput
  | GitInput;

export interface AgentInput {
  /**
   * The natural language instruction for the agent
   */
  prompt: string;
  /**
   * List of file paths to provide as context to the agent
   */
  context_files?: string[];
}
export interface BashInput {
  /**
   * The command to execute
   */
  command: string;
  /**
   * Optional timeout in milliseconds (max 600000)
   */
  timeout?: number;
  /**
   * Clear, concise description of what this command does in 5-10 words, in active voice. Examples:
   * Input: ls
   * Output: List files in current directory
   *
   * Input: git status
   * Output: Show working tree status
   *
   * Input: npm install
   * Output: Install package dependencies
   *
   * Input: mkdir foo
   * Output: Create directory 'foo'
   */
  description?: string;
  /**
   * Set to true to run this command in the background. Use TaskOutput to read the output later.
   */
  run_in_background?: boolean;
  /**
   * Set this to true to dangerously override sandbox mode and run commands without sandboxing.
   */
  dangerouslyDisableSandbox?: boolean;
}
export interface FileEditInput {
  /**
   * The absolute path to the file to edit (must be absolute, not relative)
   */
  path: string;
  /**
   * The text to search for and replace
   */
  old_string: string;
  /**
   * The text to replace the old_string with
   */
  new_string: string;
  /**
   * Whether to replace all occurrences of old_string or just the first one
   */
  replace_all?: boolean;
}
export interface FileReadInput {
  /**
   * The absolute path to the file to read
   */
  file_path: string;
  /**
   * The line number to start reading from. Only provide if the file is too large to read at once
   */
  offset?: number;
  /**
   * The number of lines to read. Only provide if the file is too large to read at once.
   */
  limit?: number;
}
export interface FileWriteInput {
  /**
   * The absolute path to the file to write (must be absolute, not relative)
   */
  file_path: string;
  /**
   * The content to write to the file
   */
  content: string;
}
export interface GrepInput {
  /**
   * The regular expression pattern to search for in file contents
   */
  pattern: string;
  /**
   * File or directory to search in (rg PATH). Defaults to current working directory.
   */
  path?: string;
  /**
   * Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob
   */
  glob?: string;
  /**
   * Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "files_with_matches" shows file paths (supports head_limit), "count" shows match counts (supports head_limit). Defaults to "files_with_matches".
   */
  output_mode?: "content" | "files_with_matches" | "count";
  /**
   * Number of lines to show before each match (rg -B). Requires output_mode: "content", ignored otherwise.
   */
  "-B"?: number;
  /**
   * Number of lines to show after each match (rg -A). Requires output_mode: "content", ignored otherwise.
   */
  "-A"?: number;
  /**
   * Number of lines to show before and after each match (rg -C). Requires output_mode: "content", ignored otherwise.
   */
  "-C"?: number;
  /**
   * Show line numbers in output (rg -n). Requires output_mode: "content", ignored otherwise. Defaults to true.
   */
  "-n"?: boolean;
  /**
   * Case insensitive search (rg -i)
   */
  "-i"?: boolean;
  /**
   * File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types.
   */
  type?: string;
  /**
   * Limit output to first N lines/entries, equivalent to "| head -N". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 0 (unlimited).
   */
  head_limit?: number;
  /**
   * Skip first N lines/entries before applying head_limit, equivalent to "| tail -n +N | head -N". Works across all output modes. Defaults to 0.
   */
  offset?: number;
  /**
   * Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.
   */
  multiline?: boolean;
}
export interface FileSearchInput {
  /**
   * The glob pattern to match files against
   */
  pattern: string;
  /**
   * The directory to search in. If not specified, the current working directory will be used.
   */
  path?: string;
}
export interface TaskOutputInput {
  /**
   * The ID of the background task to read output from
   */
  taskId: string;
}
export interface KillShellInput {
  /**
   * The ID of the background task/shell to kill
   */
  taskId: string;
}
export interface AskUserQuestionInput {
  /**
   * 1-4 questions
   *
   * @minItems 1
   * @maxItems 4
   */
  questions:
    | [
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        }
      ]
    | [
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        },
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        }
      ]
    | [
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        },
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        },
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        }
      ]
    | [
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        },
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        },
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        },
        {
          /**
           * The question to ask
           */
          question: string;
          /**
           * Short label/chip name
           */
          header: string;
          /**
           * 2-4 options
           */
          options:
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ]
            | [
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                },
                {
                  /**
                   * Concise display text (1-5 words)
                   */
                  label: string;
                  /**
                   * Explanation context
                   */
                  description: string;
                }
              ];
          /**
           * Allow multiple selection
           */
          multiSelect: boolean;
        }
      ];
  /**
   * User answers collected by UI
   */
  answers?: {
    [k: string]: string;
  };
}
export interface NotebookEditInput {
  /**
   * Absolute path to the notebook file (must be absolute, not relative)
   */
  notebook_path: string;
  /**
   * ID of the cell to edit
   */
  cell_id?: string;
  /**
   * New content for the cell
   */
  new_source?: string;
  /**
   * Type of the cell
   */
  cell_type?: "code" | "markdown";
  /**
   * Mode of operation
   */
  edit_mode: "replace" | "insert" | "delete";
}
export interface McpInput {
  /**
   * The name of the MCP server
   */
  server_name: string;
  /**
   * The name of the tool to call
   */
  tool_name: string;
  /**
   * Arguments to pass to the tool
   */
  arguments: {
    [k: string]: unknown;
  };
}
export interface WebSearchInput {
  /**
   * The search query to use
   */
  query: string;
  /**
   * Only include search results from these domains
   */
  allowed_domains?: string[];
  /**
   * Never include search results from these domains
   */
  blocked_domains?: string[];
}
export interface GitInput {
  /**
   * Git action to perform (e.g. status, log, diff)
   */
  action: string;
  /**
   * Arguments for the git action
   */
  args?: string[];
}
