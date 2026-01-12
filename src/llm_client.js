const { GoogleGenerativeAI } = require('@google/generative-ai');
const { OpenRouter } = require('@openrouter/sdk');
const dotenv = require('dotenv');
dotenv.config();

const PROVIDER = process.env.LLM_PROVIDER || 'gemini';
const MODEL = process.env.LLM_MODEL || (PROVIDER === 'gemini' ? 'gemini-2.0-flash' : 'google/gemini-2.0-flash-exp:free');
const API_KEY = PROVIDER === 'gemini' ? process.env.GEMINI_API_KEY : process.env.OPENROUTER_API_KEY;

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
            const result = await model.generateContent(prompt);
            const response = await result.response;
            return response.text();
        } else {
            // Official OpenRouter SDK
            const response = await openrouterClient.chat.send({
                model: MODEL,
                messages: [{ role: 'user', content: prompt }],
                // Note: response_format handling might differ in the beta SDK, 
                // but usually stays consistent with the underlying API.
            });

            if (!response.choices || !response.choices[0].message) {
                throw new Error(`Unexpected OpenRouter response format: ${JSON.stringify(response)}`);
            }
            return response.choices[0].message.content;
        }
    } catch (err) {
        // SDKs often have their own error structures
        const status = err.status || (err.response ? err.response.status : null);

        // Handle Rate Limiting (429) or Server Errors (5xx) with retry
        if ((status === 429 || (status >= 500 && status < 600)) && retryCount < 7) {
            const delay = Math.pow(2, retryCount) * 2000 + Math.random() * 1000;
            console.warn(`[!] ${status === 429 ? 'Rate limited (429)' : `Server error (${status})`}. Retrying in ${Math.round(delay / 1000)}s (Attempt ${retryCount + 1}/7)...`);
            await sleep(delay);
            return callLLM(prompt, retryCount + 1);
        }

        // Handle specific API key errors
        if (status === 401 || status === 403) {
            throw new Error(`[!] API Key error (${status}): Authentication failed. Please check your ${PROVIDER === 'gemini' ? 'GEMINI_API_KEY' : 'OPENROUTER_API_KEY'}.`);
        }

        throw err;
    }
}

module.exports = {
    callLLM,
    PROVIDER,
    MODEL
};
