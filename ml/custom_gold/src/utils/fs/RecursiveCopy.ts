import * as fs from "fs";
import * as path from "path";

export function copyRecursiveSync(src: string, dest: string) {
    if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
    if (!fs.existsSync(src)) return; // Should probably throw or warn

    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);

        if (entry.isDirectory()) {
            copyRecursiveSync(srcPath, destPath);
        } else if (entry.isFile()) {
            fs.copyFileSync(srcPath, destPath);
        } else if (entry.isSymbolicLink()) {
            const target = fs.readlinkSync(srcPath);
            // Logic to handle symlinks safely (from qPA)
            try {
                fs.symlinkSync(target, destPath);
            } catch (e) {
                // Ignore if exists or specific error
            }
        }
    }
}
