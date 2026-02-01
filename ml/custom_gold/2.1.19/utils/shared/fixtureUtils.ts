/**
 * File: src/utils/shared/fixtureUtils.ts
 * Role: Utilities for recording and playing back API responses (fixtures).
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { createHash } from "node:crypto";

import { EnvService } from "../../services/config/EnvService.js";

const FIXTURES_ROOT = EnvService.get("CLAUDE_CODE_TEST_FIXTURES_ROOT");

/**
 * Generates a stable key for an input object.
 */
export function generateFixtureKey(input: any): string {
    const str = typeof input === "string" ? input : JSON.stringify(input);
    return createHash("sha1").update(str).digest("hex").slice(0, 12);
}

/**
 * Wraps an operation with fixture support.
 */
export async function withFixtures<T>(key: string, operation: () => Promise<T>): Promise<T> {
    if (!EnvService.isTruthy("CLAUDE_CODE_USE_FIXTURES")) {
        return operation();
    }

    const fixturePath = join(FIXTURES_ROOT, `${key}.json`);

    if (existsSync(fixturePath)) {
        try {
            return JSON.parse(readFileSync(fixturePath, "utf-8"));
        } catch (error) {
            console.error(`[Fixtures] Failed to read fixture at ${fixturePath}:`, error);
        }
    }

    if (process.env.CI) {
        throw new Error(`[Fixtures] Fixture missing in CI: ${fixturePath}`);
    }

    const result = await operation();

    try {
        const dir = dirname(fixturePath);
        if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
        writeFileSync(fixturePath, JSON.stringify(result, null, 2), "utf-8");
    } catch (error) {
        console.error(`[Fixtures] Failed to save fixture at ${fixturePath}:`, error);
    }

    return result;
}
