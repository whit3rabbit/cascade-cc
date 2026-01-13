const { GoogleGenerativeAI } = require('@google/generative-ai');
const { OpenRouter } = require('@openrouter/sdk');
const dotenv = require('dotenv');
dotenv.config();

const PROVIDER = process.env.LLM_PROVIDER || 'gemini';
const MODEL = process.env.LLM_MODEL || (PROVIDER === 'gemini' ? 'gemini-2.0-flash' : 'google/gemini-2.0-flash-exp:free');
const API_KEY = PROVIDER === 'gemini' ? process.env.GEMINI_API_KEY : process.env.OPENROUTER_API_KEY;
const LLM_TIMEOUT_MS = 90000; // 90 seconds

let geminiClient = null;
let openaiClient = null; // Renamed to openrouterClient below for clarity if needed, but keeping variable name for minimal diff if preferred. Actually let's rename for correctness.
let openrouterClient = null;

if (PROVIDER === 'gemini' && API_KEY && !API_KEY.includes('your_')) {
    geminiClient = new GoogleGenerativeAI(API_KEY);
} else if (PROVIDER === 'openrouter' && API_KEY && !API_KEY.includes('your_')) {
    openrouterClient = new OpenRouter({
        apiKey: API_KEY,
        defaultHeaders: {
            'HTTP-Referer': 'https://github.com/whit3rabbit/cascade-like',
            'X-Title': 'Cascade-Like Deobfuscator',
        }
    });
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function callLLM(prompt, retryCount = 0) {
    if (!API_KEY || API_KEY.includes('your_')) {
        throw new Error(`[!] Missing or invalid API key for ${PROVIDER}. Please check your .env file.`);
    }

    try {
        if (PROVIDER === 'gemini') {
            const model = geminiClient.getGenerativeModel({ model: MODEL });

            // Timeout wrapper for Gemini
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error(`[!] Gemini request timed out after ${LLM_TIMEOUT_MS / 1000}s`)), LLM_TIMEOUT_MS);
            });

            const result = await Promise.race([
                model.generateContent(prompt),
                timeoutPromise
            ]);

            const response = await result.response;
            return response.text();
        } else {
            console.log(`[*] ${PROVIDER} Request: Sending prompt to ${MODEL}...`);
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);

            // Using fetch directly for OpenRouter to ensure we see the full error body
            const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
                method: "POST",
                signal: controller.signal,
                headers: {
                    "Authorization": `Bearer ${API_KEY}`,
                    "HTTP-Referer": "https://github.com/whit3rabbit/cascade-like",
                    "X-Title": "Cascade-Like Deobfuscator",
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    model: MODEL,
                    messages: [{ role: 'user', content: prompt }]
                })
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorBody = await response.json().catch(() => ({}));
                const status = response.status;
                let detailedMessage = errorBody.error ? errorBody.error.message : JSON.stringify(errorBody);

                if (status === 401 || status === 403) {
                    throw new Error(`[!] API Key error (${status}): ${detailedMessage}. Please check your OPENROUTER_API_KEY in .env.`);
                }
                if (status === 429) {
                    const isQuota = detailedMessage.toLowerCase().includes('quota') || detailedMessage.toLowerCase().includes('credit');
                    if (isQuota) throw new Error(`[!] Quota/Credit Exceeded (429): ${detailedMessage}`);

                    if (retryCount < 10) { // Increased retries for free tier
                        const delay = Math.pow(2, retryCount) * 3000 + Math.random() * 2000;
                        console.warn(`[!] Rate limited (429): ${detailedMessage}. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${retryCount + 1})`);
                        await sleep(delay);
                        return callLLM(prompt, retryCount + 1);
                    }
                }
                throw new Error(`OpenRouter error (${status}): ${detailedMessage}`);
            }

            const data = await response.json();
            if (!data.choices || !data.choices[0].message) {
                throw new Error(`Unexpected OpenRouter response format: ${JSON.stringify(data)}`);
            }
            return data.choices[0].message.content;
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            throw new Error(`[!] ${PROVIDER} request timed out after ${LLM_TIMEOUT_MS / 1000}s`);
        }
        throw err;
    }
}

async function validateKey() {
    console.log(`[*] Validating API Key for ${PROVIDER} (${MODEL})...`);
    try {
        // Simple health check request
        await callLLM("Respond with 'OK' and nothing else.");
        console.log(`[+] API Key validated successfully.`);
        return true;
    } catch (err) {
        console.error(`\n[!] API KEY VALIDATION FAILED:`);
        console.error(`    ${err.message}`);
        return false;
    }
}

module.exports = {
    callLLM,
    validateKey,
    PROVIDER,
    MODEL
};
