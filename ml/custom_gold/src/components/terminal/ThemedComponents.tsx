
import * as React from "react";
import { Text as VendorText, TextProps } from "../../vendor/inkText.js";
import { ThemedBox as VendorThemedBox } from "../../vendor/inkThemedBox.js";
import { StyledText as VendorStyledText } from "../../vendor/inkStyledText.js";
import { Link as VendorLink } from "../../vendor/inkLink.js";

/**
 * These components are core UI elements used throughout the Claude CLI.
 * They represent the deobfuscated components C, T, _3, and X9 found in various chunks.
 */

/**
 * Themed text component. 
 * Maps to 'C' in obfuscated code (e.g., chunk_202.ts).
 */
export const ThemedText = VendorText;

/**
 * Themed box component with border color support.
 * Maps to 'T' in obfuscated code (e.g., chunk_204.ts).
 */
export const ThemedBox = VendorThemedBox;

/**
 * Component that processes ANSI escape codes in text.
 * Maps to '_3' in obfuscated code (e.g., chunk_205.ts).
 */
export const StyledText = VendorStyledText;

/**
 * Terminal link component with support for 'ink-link'.
 * Maps to 'X9' in obfuscated code (e.g., chunk_204.ts).
 */
export const Link = VendorLink;

// Re-export type for convenience
export type { TextProps };
