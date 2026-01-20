import { imageProcessor } from "./imageProcessor.js";

const MAX_IMAGE_BYTES = 3932160; // 3.75 MB
const MAX_DIMENSION = 2000;

interface OptimizationResult {
    base64: string;
    mediaType: string;
    originalSize: number;
}

interface ResizeResult {
    buffer: Buffer;
    mediaType: string;
    dimensions?: {
        originalWidth: number;
        originalHeight: number;
        displayWidth: number;
        displayHeight: number;
    };
}

/**
 * Resizes an image to fit within terminal/display constraints.
 * Deobfuscated from fYA in chunk_215.ts.
 */
export async function resizeImage(
    buffer: Buffer,
    size: number,
    fallbackMediaType: string
): Promise<ResizeResult> {
    try {
        const img = imageProcessor(buffer);
        const meta = await img.metadata();
        const format = meta.format ?? fallbackMediaType;
        const mediaType = format === "jpg" ? "jpeg" : format;

        if (!meta.width || !meta.height) {
            if (size > MAX_IMAGE_BYTES) {
                return {
                    buffer: await img.jpeg({ quality: 80 }).toBuffer(),
                    mediaType: "jpeg"
                };
            }
            return { buffer, mediaType };
        }

        const { width: origW, height: origH } = meta;
        let displayW = origW;
        let displayH = origH;

        if (size <= MAX_IMAGE_BYTES && displayW <= MAX_DIMENSION && displayH <= MAX_DIMENSION) {
            return {
                buffer,
                mediaType,
                dimensions: {
                    originalWidth: origW,
                    originalHeight: origH,
                    displayWidth: displayW,
                    displayHeight: displayH
                }
            };
        }

        if (displayW > MAX_DIMENSION) {
            displayH = Math.round(displayH * MAX_DIMENSION / displayW);
            displayW = MAX_DIMENSION;
        }
        if (displayH > MAX_DIMENSION) {
            displayW = Math.round(displayW * MAX_DIMENSION / displayH);
            displayH = MAX_DIMENSION;
        }

        const resizedBuffer = await img.resize(displayW, displayH, {
            fit: "inside",
            withoutEnlargement: true
        }).toBuffer();

        if (resizedBuffer.length > MAX_IMAGE_BYTES) {
            return {
                buffer: await img.jpeg({ quality: 80 }).toBuffer(),
                mediaType: "jpeg",
                dimensions: {
                    originalWidth: origW,
                    originalHeight: origH,
                    displayWidth: displayW,
                    displayHeight: displayH
                }
            };
        }

        return {
            buffer: resizedBuffer,
            mediaType,
            dimensions: {
                originalWidth: origW,
                originalHeight: origH,
                displayWidth: displayW,
                displayHeight: displayH
            }
        };
    } catch (e) {
        return {
            buffer,
            mediaType: fallbackMediaType === "jpg" ? "jpeg" : fallbackMediaType
        };
    }
}

/**
 * Multi-pass image optimization to fit within a byte limit.
 * Deobfuscated from Y0A in chunk_215.ts.
 */
export async function optimizeImage(
    buffer: Buffer,
    maxBytes: number = MAX_IMAGE_BYTES,
    fallbackMediaType?: string
): Promise<OptimizationResult> {
    const format = fallbackMediaType?.split("/")[1] || "jpeg";
    const mediaType = format === "jpg" ? "jpeg" : format;

    try {
        const img = imageProcessor(buffer);
        const meta = await img.metadata();
        const currentFormat = meta.format || mediaType;
        const currentSize = buffer.length;

        if (currentSize <= maxBytes) {
            return {
                base64: buffer.toString("base64"),
                mediaType: `image/${currentFormat === "jpg" ? "jpeg" : currentFormat}`,
                originalSize: currentSize
            };
        }

        // Pass 1: Resize iteratively
        const scales = [1, 0.75, 0.5, 0.25];
        for (const scale of scales) {
            const w = Math.round((meta.width || 2000) * scale);
            const h = Math.round((meta.height || 2000) * scale);
            let op = imageProcessor(buffer).resize(w, h, {
                fit: "inside",
                withoutEnlargement: true
            });

            // Apply format-specific optimization
            if (currentFormat === "png") {
                op = op.png({ compressionLevel: 9, palette: true });
            } else if (currentFormat === "jpeg" || currentFormat === "jpg") {
                op = op.jpeg({ quality: 80 });
            } else if (currentFormat === "webp") {
                op = op.webp({ quality: 80 });
            }

            const optimized = await op.toBuffer();
            if (optimized.length <= maxBytes) {
                return {
                    base64: optimized.toString("base64"),
                    mediaType: `image/${currentFormat === "jpg" ? "jpeg" : currentFormat}`,
                    originalSize: currentSize
                };
            }
        }

        // Pass 2: Extreme PNG optimization
        if (currentFormat === "png") {
            const smallPng = await imageProcessor(buffer).resize(800, 800, {
                fit: "inside",
                withoutEnlargement: true
            }).png({
                compressionLevel: 9,
                palette: true,
                colors: 64
            }).toBuffer();
            if (smallPng.length <= maxBytes) {
                return {
                    base64: smallPng.toString("base64"),
                    mediaType: "image/png",
                    originalSize: currentSize
                };
            }
        }

        // Pass 3: Low quality JPEG
        const lowJpeg = await imageProcessor(buffer).resize(600, 600, {
            fit: "inside",
            withoutEnlargement: true
        }).jpeg({ quality: 50 }).toBuffer();
        if (lowJpeg.length <= maxBytes) {
            return {
                base64: lowJpeg.toString("base64"),
                mediaType: "image/jpeg",
                originalSize: currentSize
            };
        }

        // Pass 4: Tiny JPEG (last resort)
        const tinyJpeg = await imageProcessor(buffer).resize(400, 400, {
            fit: "inside",
            withoutEnlargement: true
        }).jpeg({ quality: 20 }).toBuffer();
        return {
            base64: tinyJpeg.toString("base64"),
            mediaType: "image/jpeg",
            originalSize: currentSize
        };

    } catch (e) {
        return {
            base64: buffer.toString("base64"),
            mediaType: `image/${mediaType}`,
            originalSize: buffer.length
        };
    }
}
