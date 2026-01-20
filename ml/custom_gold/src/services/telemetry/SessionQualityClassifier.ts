
const MAX_MESSAGE_CONTENT_LENGTH = 300;

function extractUserMessages(messages: any[]) {
    const result: string[] = [];
    for (const msg of messages) {
        if (msg.type === "user" && msg.message?.content) {
            let content = "";
            if (typeof msg.message.content === "string") {
                content = msg.message.content;
            } else if (Array.isArray(msg.message.content)) {
                for (const part of msg.message.content) {
                    if (part.type === "text") {
                        content += part.text + " ";
                    }
                }
            }
            if (content.trim()) {
                result.push(content.trim().slice(0, MAX_MESSAGE_CONTENT_LENGTH));
            }
        }
    }
    return result;
}

function formatConversationForAnalysis(userMessages: string[]) {
    return userMessages.map((msg) => `User: ${msg}\nAsst: [response hidden]`).join("\n");
}

function parseQualityResult(response: string) {
    const frustratedMatch = response.match(/<frustrated>(true|false)<\/frustrated>/);
    const prRequestMatch = response.match(/<pr_request>(true|false)<\/pr_request>/);

    return {
        isFrustrated: frustratedMatch ? frustratedMatch[1] === "true" : false,
        hasPRRequest: prRequestMatch ? prRequestMatch[1] === "true" : false
    };
}

export const SessionQualityClassifier = {
    name: "session_quality_classifier",
    async shouldRun(context: any) {
        if (context.querySource !== "repl_main_thread") return false;
        return extractUserMessages(context.messages).length > 0;
    },
    buildMessages(context: any) {
        const userMessages = extractUserMessages(context.messages);
        const body = formatConversationForAnalysis(userMessages);
        return [
            {
                role: "user",
                content: `Analyze the following conversation between a user and an assistant (assistant responses are hidden).

${body}

Think step-by-step about:
1. Does the user seem frustrated at the Asst based on their messages? Look for signs like repeated corrections, negative language, etc.
2. Has the user explicitly asked to SEND/CREATE/PUSH a pull request to GitHub? This means they want to actually submit a PR to a repository, not just work on code together or prepare changes. Look for explicit requests like: "create a pr", "send a pull request", "push a pr", "open a pr", "submit a pr to github", etc. Do NOT count mentions of working on a PR together, preparing for a PR, or discussing PR content.

Based on your analysis, output:
<frustrated>true/false</frustrated>
<pr_request>true/false</pr_request>`
            }
        ];
    },
    systemPrompt: "You are analyzing user messages from a conversation to detect certain features of the interaction.",
    useTools: false,
    parseResponse(response: string) {
        return parseQualityResult(response);
    }
};
