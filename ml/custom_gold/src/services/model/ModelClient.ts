import { callClaude } from "../claude/claudeApi.js";

export function getModelClient() {
    return {
        complete: async (params: any) => {
            const { messages, system, signal, model, max_tokens, tools, tool_choice, temperature } = params;

            const response = await callClaude({
                messages,
                systemPrompt: system,
                signal,
                model,
                max_tokens,
                tools,
                tool_choice,
                temperature
            });

            // Extract text content from the response
            let text = "";
            if (response.content) {
                const textBlock = response.content.find((c: any) => c.type === "text");
                if (textBlock) {
                    text = textBlock.text;
                }
            }

            return {
                text,
                ...response
            };
        }
    };
}
