
export interface Tool {
    name: string;
    description: (input?: any) => Promise<string> | string;
    prompt: (input?: any) => Promise<string> | string;
    inputSchema: any;
    outputSchema?: any;
    examples?: any[];
    input_examples?: any[];
    strict?: boolean;
    userFacingName: (input: any) => string;
    getToolUseSummary?: (input: any) => string;
    isEnabled: (context?: any) => boolean;
    isConcurrencySafe: (context?: any) => boolean;
    isReadOnly: (context?: any) => boolean;
    isSearchOrReadCommand: (context?: any) => { isSearch: boolean; isRead: boolean; };
    getPath?: (input: any) => string;
    checkPermissions?: (input: any, context: any) => Promise<boolean>;
    validateInput?: (input: any, context: any) => Promise<{ result: boolean; message?: string; errorCode?: number }>;
    call: (input: any, context: any) => Promise<any>;

    renderToolUseMessage?: (input: any, options: any) => any;
    renderToolUseTag?: (input: any) => any;
    renderToolUseProgressMessage?: () => any;
    renderToolResultMessage?: (result: any) => any;
    renderToolUseRejectedMessage?: () => any;
    renderToolUseErrorMessage?: (error: any) => any;
    mapToolResultToToolResultBlockParam?: (result: any, toolUseId: string) => any;
}
