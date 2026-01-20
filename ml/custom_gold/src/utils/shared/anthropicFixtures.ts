import fs from "fs";
import path from "path";
import crypto from "crypto";

// Mocking or assuming these exist in the codebase
// These will be resolved during deobfuscation of other chunks or are already in deobfuscation.md
const isFixturesEnabled = () => process.env.CLAUDE_CODE_USE_FIXTURES === "true";
const getCwd = () => process.env.CLAUDE_CODE_TEST_FIXTURES_ROOT ?? process.cwd();
const getConfigHome = () => process.env.CLAUDE_CODE_CONFIG_HOME ?? "";

/**
 * Normalizes text by replacing dynamic values (like durations, costs, counts) 
 * with placeholders to ensure stable fixture hashes.
 */
export function normalizeText(text: string): string {
    if (typeof text !== "string") return text;
    return text
        .replace(/num_files="\d+"/g, 'num_files="[NUM]"')
        .replace(/duration_ms="\d+"/g, 'duration_ms="[DURATION]"')
        .replace(/cost_usd="\d+"/g, 'cost_usd="[COST]"')
        .replace(/\//g, path.sep)
        .replaceAll(getConfigHome(), "[CONFIG_HOME]")
        .replaceAll(process.cwd(), "[CWD]")
        .replace(/Available commands:.+/, "Available commands: [COMMANDS]")
        .replace(/Files modified by user:.+/, "Files modified by user: [FILES]");
}

/**
 * Restores placeholders in text back to their actual values if possible,
 * or to fixed dummy values for consistent output in tests.
 */
export function restoreText(text: string): string {
    if (typeof text !== "string") return text;
    return text
        .replaceAll("[NUM]", "1")
        .replaceAll("[DURATION]", "100")
        .replaceAll("[CONFIG_HOME]", getConfigHome())
        .replaceAll("[CWD]", process.cwd());
}

function transformContent(content: any, transformer: (s: string) => string): any {
    if (typeof content === "string") return transformer(content);
    if (Array.isArray(content)) {
        return content.map(item => {
            if (item.type === "text") return { ...item, text: transformer(item.text) };
            if (item.type === "tool_result") {
                if (typeof item.content === "string") return { ...item, content: transformer(item.content) };
                if (Array.isArray(item.content)) {
                    return {
                        ...item,
                        content: item.content.map((c: any) => c.type === "text" ? { ...c, text: transformer(c.text) } : c)
                    };
                }
            }
            if (item.type === "tool_use") {
                return { ...item, input: transformInput(item.input, transformer) };
            }
            return item;
        });
    }
    return content;
}

function transformInput(input: any, transformer: (s: string) => string): any {
    if (Array.isArray(input)) return input.map(i => transformInput(i, transformer));
    if (typeof input === "object" && input !== null) {
        const result: any = {};
        for (const key in input) {
            result[key] = transformInput(input[key], transformer);
        }
        return result;
    }
    if (typeof input === "string") return transformer(input);
    return input;
}

function anonymizeAssistantMessage(message: any, transformer: (s: string) => string, index: number): any {
    if (message.type !== "assistant") return message;
    return {
        uuid: `UUID-${index}`,
        requestId: "REQUEST_ID",
        timestamp: message.timestamp,
        message: {
            ...message.message,
            content: (Array.isArray(message.message.content) ? message.message.content : [message.message.content])
                .map((item: any) => {
                    if (item.type === "text") return { ...item, text: transformer(item.text), citations: item.citations || [] };
                    if (item.type === "tool_use") return { ...item, input: transformInput(item.input, transformer) };
                    return item;
                })
        },
        type: "assistant"
    };
}

/**
 * Higher-order function to wrap Anthropic API calls with fixture support.
 * Useful for deterministic testing and avoiding actual API calls in certain environments.
 */
export async function withAnthropicFixtures<T>(
    messages: any[],
    callApi: () => Promise<T>
): Promise<T> {
    if (!isFixturesEnabled()) return await callApi();

    const filteredMessages = messages.filter(m => m.type !== "user" || !m.isMeta);
    const normalizedInput = filteredMessages.map(m => transformContent(m.message.content, normalizeText));

    const hash = crypto.createHash("sha1")
        .update(JSON.stringify(normalizedInput))
        .digest("hex")
        .slice(0, 6);

    const fixturesRoot = process.env.CLAUDE_CODE_TEST_FIXTURES_ROOT ?? process.cwd();
    const fixturePath = path.join(fixturesRoot, `fixtures/${hash}.json`);

    if (fs.existsSync(fixturePath)) {
        const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
        return fixture.output.map((out: any, i: number) => anonymizeAssistantMessage(out, restoreText, i)) as T;
    }

    if (process.env.CI) {
        throw new Error(`Anthropic API fixture missing: ${fixturePath}. Re-run tests locally to generate. Input: ${JSON.stringify(normalizedInput, null, 2)}`);
    }

    const output = await callApi();

    if (!fs.existsSync(path.dirname(fixturePath))) {
        fs.mkdirSync(path.dirname(fixturePath), { recursive: true });
    }

    fs.writeFileSync(fixturePath, JSON.stringify({
        input: normalizedInput,
        output: Array.isArray(output) ? output.map((out, i) => anonymizeAssistantMessage(out, normalizeText, i)) : output
    }, null, 2), "utf8");

    return output;
}

/**
 * Async generator version for streaming responses.
 */
export async function* withAnthropicFixturesStream<T>(
    messages: any[],
    streamApi: () => AsyncGenerator<T>
): AsyncGenerator<T> {
    if (!isFixturesEnabled()) {
        yield* streamApi();
        return;
    }

    const chunks: T[] = [];
    const fullResult = await withAnthropicFixtures(messages, async () => {
        for await (const chunk of streamApi()) {
            chunks.push(chunk);
        }
        return chunks;
    });

    if (fullResult.length > 0) {
        yield* (fullResult as any);
    } else {
        yield* chunks;
    }
}
