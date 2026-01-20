
/**
 * Facade for file and command suggestions.
 */

export {
    suggestFiles,
    suggestDirectories,
    refreshFileIndex as updateFileIndex
} from '../file-index/FileIndexService.js';

export {
    getSlashCommandSuggestions as suggestSlashCommands
} from '../../utils/terminal/Suggestions.js';

import { wrapCommandWithNoStdin } from "../../utils/terminal/ShellUtils.js";

/**
 * Escapes and redirects stdin for shell commands to prevent them from
 * hanging or taking over the terminal input.
 */
export function wrapShellCommand(command: string, redirectStdin = true): string {
    return wrapCommandWithNoStdin(command, redirectStdin);
}
