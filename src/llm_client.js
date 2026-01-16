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

const PROVIDER_CONFIG = {
    gemini: {
        delay: 4000, // 4s flat delay for Gemini free tier reliability
        retries: 3
    },
    openrouter: {
        delay: 1000,
        retries: 5
    }
};

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
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);

            try {
                const result = await model.generateContent({
                    contents: [{ role: 'user', parts: [{ text: prompt }] }],
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                const response = await result.response;
                return response.text();
            } catch (err) {
                clearTimeout(timeoutId);

                // Retry for Gemini transient errors (e.g. 503, connectivity)
                const isTransient = err.message?.includes('503') || err.message?.includes('500') || err.message?.includes('429');
                const isNetwork = err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT' || err.message?.includes('fetch failed');

                if ((isTransient || isNetwork) && retryCount < 5) {
                    const delay = Math.pow(2, retryCount) * 2000 + Math.random() * 1000;
                    console.warn(`[!] Gemini Error: ${err.message}. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${retryCount + 1})`);
                    await sleep(delay);
                    return callLLM(prompt, retryCount + 1);
                }
                throw err;
            }
        } else {
            console.log(`[*] ${PROVIDER} Request: Sending prompt to ${MODEL}...`);
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);

            let response;
            try {
                response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
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
            } catch (fetchErr) {
                clearTimeout(timeoutId);
                const isNetwork = fetchErr.code === 'ECONNRESET' || fetchErr.code === 'ETIMEDOUT' || fetchErr.message?.includes('fetch failed') || fetchErr.name === 'AbortError';
                if (isNetwork && retryCount < 5) {
                    const delay = Math.pow(2, retryCount) * 2000 + Math.random() * 1000;
                    console.warn(`[!] OpenRouter Network Error: ${fetchErr.message}. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${retryCount + 1})`);
                    await sleep(delay);
                    return callLLM(prompt, retryCount + 1);
                }
                throw fetchErr;
            }

            clearTimeout(timeoutId);

            if (!response.ok) {
                const status = response.status;
                let errorBody = {};
                try {
                    errorBody = await response.json();
                } catch (e) {
                    errorBody = { message: "Could not parse error response" };
                }

                let detailedMessage = errorBody.error ? errorBody.error.message : JSON.stringify(errorBody);

                // Check for insufficient credits
                const isInsufficientCredits = detailedMessage.toLowerCase().includes('insufficient credits') || detailedMessage.toLowerCase().includes('credit balance');
                if (isInsufficientCredits) {
                    console.error(`[!] OpenRouter Error: Insufficient Credits. Please top up your account.`);
                    throw new Error("OPENROUTER_INSUFFICIENT_CREDITS");
                }

                if (status === 401 || status === 403) {
                    throw new Error(`[!] API Key error (${status}): ${detailedMessage}. Please check your OPENROUTER_API_KEY in .env.`);
                }

                // Retry for Rate Limited (429) OR Transient Server Errors (5xx)
                if (status === 429 || (status >= 500 && status <= 504)) {
                    if (retryCount < (status === 429 ? 10 : 5)) {
                        const delay = Math.pow(2, retryCount) * 3000 + Math.random() * 2000;
                        console.warn(`[!] ${status === 429 ? 'Rate limited' : 'Server error'} (${status}): ${detailedMessage}. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${retryCount + 1})`);
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
        if (err.name === 'AbortError' && retryCount < 5) {
            const delay = Math.pow(2, retryCount) * 2000 + Math.random() * 1000;
            console.warn(`[!] Request timed out. Retrying in ${Math.round(delay / 1000)}s... (Attempt ${retryCount + 1})`);
            await sleep(delay);
            return callLLM(prompt, retryCount + 1);
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
