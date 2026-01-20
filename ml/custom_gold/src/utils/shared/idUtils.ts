import { randomBytes } from "crypto";

const ADJECTIVES = [
    "swift", "silent", "ancient", "vibrant", "mighty", "golden", "shadowy", "azure", "crimson", "frosty"
];

const NOUNS = [
    "falcon", "river", "mountain", "forest", "shield", "blade", "storm", "valley", "star", "phoenix"
];

const VERBS = [
    "soars", "flows", "stands", "glows", "guards", "cuts", "rages", "sleeps", "shines", "rises"
];

function getRandomInt(max: number): number {
    return randomBytes(4).readUInt32BE(0) % max;
}

function getRandomItem<T>(list: T[]): T {
    return list[getRandomInt(list.length)];
}

/**
 * Generates a random 3-word slug (e.g., "swift-river-soars").
 */
export function generateSlug(): string {
    const adj = getRandomItem(ADJECTIVES);
    const noun = getRandomItem(NOUNS);
    const verb = getRandomItem(VERBS);
    return `${adj}-${noun}-${verb}`;
}
