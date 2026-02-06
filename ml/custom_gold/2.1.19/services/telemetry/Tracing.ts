/**
 * File: src/services/telemetry/Tracing.ts
 * Role: Tracing Utilities - Context, Baggage, and Attribute Management
 */

import { createContextKey, diag } from '../../utils/shared/runtime.js';
import { baggageEntryMetadataFromString, propagation, createBaggage } from '../../utils/shared/runtimeAndEnv.js';

// --- Types ---

interface BaggageEntry {
    value: string;
    metadata?: any;
}

interface Context {
    setValue(key: symbol, value: any): Context;
    deleteValue(key: symbol): Context;
    getValue(key: symbol): any;
}

// --- Constants ---
const SUPPRESS_TRACING_CONTEXT_KEY = createContextKey("OpenTelemetry SDK Context Key SUPPRESS_TRACING");
const BAGGAGE_KEY_PAIR_SEPARATOR = "=";
const BAGGAGE_PROPERTIES_SEPARATOR = ";";
const BAGGAGE_ITEMS_SEPARATOR = ",";
const BAGGAGE_HEADER = "baggage";
const BAGGAGE_MAX_NAME_VALUE_PAIRS = 180;
const BAGGAGE_MAX_PER_NAME_VALUE_PAIRS = 4096;
const BAGGAGE_MAX_TOTAL_LENGTH = 8192;

// --- Context Management ---

export function suppressTracing(context: Context): Context {
    return context.setValue(SUPPRESS_TRACING_CONTEXT_KEY, true);
}

export function unsuppressTracing(context: Context): Context {
    return context.deleteValue(SUPPRESS_TRACING_CONTEXT_KEY);
}

export function isTracingSuppressed(context: Context): boolean {
    return context.getValue(SUPPRESS_TRACING_CONTEXT_KEY) === true;
}

// --- Baggage Management ---

export class W3CBaggagePropagator {
    inject(context: Context, carrier: any, setter: any): void {
        const baggage = propagation.getBaggage(context);
        if (!baggage || isTracingSuppressed(context)) {
            return;
        }

        const keyPairs = getKeyPairs(baggage)
            .filter(keyPair => keyPair.length <= BAGGAGE_MAX_PER_NAME_VALUE_PAIRS)
            .slice(0, BAGGAGE_MAX_NAME_VALUE_PAIRS);

        const serializedPairs = serializeKeyPairs(keyPairs);
        if (serializedPairs.length > 0) {
            setter.set(carrier, BAGGAGE_HEADER, serializedPairs);
        }
    }

    extract(context: Context, carrier: any, getter: any): Context {
        const baggageHeader = getter.get(carrier, BAGGAGE_HEADER);
        const baggageHeaderValue = Array.isArray(baggageHeader)
            ? baggageHeader.join(BAGGAGE_ITEMS_SEPARATOR)
            : baggageHeader;

        if (!baggageHeaderValue) {
            return context;
        }

        const baggageRecord: Record<string, BaggageEntry> = {};
        if (baggageHeaderValue.length === 0) {
            return context;
        }

        baggageHeaderValue.split(BAGGAGE_ITEMS_SEPARATOR).forEach((item: string) => {
            const parsedItem = parsePairKeyValue(item);
            if (parsedItem) {
                const entry: BaggageEntry = { value: parsedItem.value };
                if (parsedItem.metadata) {
                    entry.metadata = parsedItem.metadata;
                }
                baggageRecord[parsedItem.key] = entry;
            }
        });

        if (Object.keys(baggageRecord).length === 0) {
            return context;
        }

        return propagation.setBaggage(context, createBaggage(baggageRecord));
    }

    fields(): string[] {
        return [BAGGAGE_HEADER];
    }
}

// --- Attribute Management ---

export function sanitizeAttributes(attributes: Record<string, any>): Record<string, any> {
    const sanitized: Record<string, any> = {};
    if (typeof attributes !== "object" || attributes == null) {
        return sanitized;
    }

    for (const key in attributes) {
        if (!Object.prototype.hasOwnProperty.call(attributes, key)) {
            continue;
        }

        if (!isAttributeKey(key)) {
            diag.warn(`Invalid attribute key: ${key}`);
            continue;
        }

        const value = attributes[key];
        if (!isAttributeValue(value)) {
            diag.warn(`Invalid attribute value set for key: ${key}`);
            continue;
        }

        if (Array.isArray(value)) {
            sanitized[key] = value.slice();
        } else {
            sanitized[key] = value;
        }
    }

    return sanitized;
}

export function isAttributeKey(key: any): boolean {
    return typeof key === "string" && key !== "";
}

export function isAttributeValue(value: any): boolean {
    if (value == null) return true;
    if (Array.isArray(value)) return isArrayOfPrimitive(value);
    return isPrimitiveType(typeof value);
}

// --- Helpers ---

function serializeKeyPairs(keyPairs: string[]): string {
    return keyPairs.reduce((accumulator, currentPair) => {
        let combined = `${accumulator}${accumulator !== "" ? BAGGAGE_ITEMS_SEPARATOR : ""}${currentPair}`;
        return combined.length > BAGGAGE_MAX_TOTAL_LENGTH ? accumulator : combined;
    }, "");
}

function getKeyPairs(baggage: any): string[] {
    return baggage.getAllEntries().map(([key, entry]: [string, BaggageEntry]) => {
        let keyValueString = `${encodeURIComponent(key)}=${encodeURIComponent(entry.value)}`;
        if (entry.metadata !== undefined) {
            keyValueString += BAGGAGE_PROPERTIES_SEPARATOR + entry.metadata.toString();
        }
        return keyValueString;
    });
}

function parsePairKeyValue(keyValueString: string): { key: string, value: string, metadata?: any } | undefined {
    const parts = keyValueString.split(BAGGAGE_PROPERTIES_SEPARATOR);
    if (parts.length <= 0) return undefined;

    const firstPart = parts.shift();
    if (!firstPart) return undefined;

    const equalsIndex = firstPart.indexOf(BAGGAGE_KEY_PAIR_SEPARATOR);
    if (equalsIndex <= 0) return undefined;

    const key = decodeURIComponent(firstPart.substring(0, equalsIndex).trim());
    const value = decodeURIComponent(firstPart.substring(equalsIndex + 1).trim());

    let metadata;
    if (parts.length > 0) {
        metadata = baggageEntryMetadataFromString(parts.join(BAGGAGE_PROPERTIES_SEPARATOR));
    }

    return { key, value, metadata };
}

function isArrayOfPrimitive(arr: any[]): boolean {
    let expectedType: string | undefined;
    for (const item of arr) {
        if (item == null) continue;
        const itemType = typeof item;
        if (itemType === expectedType) continue;

        if (!expectedType) {
            if (isPrimitiveType(itemType)) {
                expectedType = itemType;
                continue;
            }
            return false;
        }
        return false;
    }
    return true;
}

function isPrimitiveType(type: string): boolean {
    return type === "number" || type === "boolean" || type === "string";
}
