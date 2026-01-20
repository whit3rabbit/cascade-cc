/**
 * Wrapper for native image processor.
 * Deobfuscated from sKB in chunk_207.ts.
 */

import { createRequire } from "module";
import path from "path";
import { fileURLToPath } from "url";

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

let nativeModule: any = null;
function loadNativeBinary() {
    const searchPaths = [
        path.join(__dirname, "../../vendor/image-processor.node"),
        path.join(__dirname, "vendor/image-processor.node"),
        path.join(__dirname, "image-processor.node"),
        path.join(process.cwd(), "vendor/image-processor.node"),
        path.join(process.cwd(), "image-processor.node"),
        path.join(process.cwd(), "binaries", "image-processor.node"),
        path.join(path.dirname(process.argv[1] ?? ""), "image-processor.node")
    ];

    for (const binPath of searchPaths) {
        try {
            // @ts-ignore
            return require(binPath);
        } catch (e) {
            continue;
        }
    }
    return null;
}

nativeModule = loadNativeBinary();

export function imageProcessor(buffer: Buffer) {
    let promise: Promise<any> | null = null;
    const operations: Array<(img: any) => void> = [];

    async function getProcessedImage() {
        if (!promise) {
            promise = (async () => {
                if (!nativeModule) {
                    throw new Error("Native image processor module not available");
                }
                const img = await nativeModule.processImage(buffer);
                for (const op of operations) {
                    op(img);
                }
                return img;
            })();
        }
        return promise;
    }

    const api = {
        async metadata() {
            const img = await getProcessedImage();
            return img.metadata();
        },

        resize(width?: number, height?: number, options?: any) {
            operations.push((img) => img.resize(width, height, options));
            return api;
        },

        jpeg(options?: { quality?: number }) {
            operations.push((img) => img.jpeg(options?.quality));
            return api;
        },

        png(options?: any) {
            operations.push((img) => img.png(options));
            return api;
        },

        webp(options?: { quality?: number }) {
            operations.push((img) => img.webp(options?.quality));
            return api;
        },

        async toBuffer() {
            const img = await getProcessedImage();
            return img.toBuffer();
        }
    };

    return api;
}

export default imageProcessor;
