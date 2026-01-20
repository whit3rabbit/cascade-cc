import * as os from "node:os";
import * as tty from "node:tty";

/**
 * Detects terminal color support level.
 * Deobfuscated from sWB in chunk_204.ts.
 */
export function supportsColor(stream: tty.WriteStream): { level: number; hasBasic: boolean; has256: boolean; has16m: boolean } | false {
    const level = getLevel(stream);
    if (level === 0) return false;
    return {
        level,
        hasBasic: true,
        has256: level >= 2,
        has16m: level >= 3
    };
}

function getLevel(stream: tty.WriteStream): number {
    const { env } = process;

    if (env.TERM === "dumb") return 0;

    if (process.platform === "win32") {
        const release = os.release().split(".");
        if (Number(release[0]) >= 10 && Number(release[2]) >= 10586) {
            return Number(release[2]) >= 14931 ? 3 : 2;
        }
        return 1;
    }

    if ("CI" in env) {
        if (["TRAVIS", "CIRCLECI", "APPVEYOR", "GITLAB_CI", "GITHUB_ACTIONS", "BUILDKITE"].some(ci => ci in env)) return 1;
        return 0;
    }

    if (env.COLORTERM === "truecolor") return 3;

    if (env.TERM_PROGRAM === "iTerm.app") {
        const version = parseInt((env.TERM_PROGRAM_VERSION || "").split(".")[0], 10);
        return version >= 3 ? 3 : 2;
    }

    if (/-256(color)?$/i.test(env.TERM || "")) return 2;
    if (/^screen|^xterm|^vt100|^vt220|^rxvt|color|ansi|cygwin|linux/i.test(env.TERM || "")) return 1;

    return stream.isTTY ? 1 : 0;
}

/**
 * Detects terminal hyperlink support.
 * Deobfuscated from kl1 in chunk_204.ts.
 */
export function supportsHyperlink(stream: tty.WriteStream): boolean {
    const { env } = process;
    const forceHyperlink = env.FORCE_HYPERLINK;
    if (forceHyperlink !== undefined) {
        return !(forceHyperlink.length > 0 && parseInt(forceHyperlink, 10) === 0);
    }

    if (env.NETLIFY) return true;

    if (!stream.isTTY) return false;
    if (process.platform === "win32") return false;

    if ("CI" in env) return false;
    if ("TEAMCITY_VERSION" in env) return false;

    if (env.TERM_PROGRAM === "iTerm.app") {
        const version = getVersion(env.TERM_PROGRAM_VERSION);
        if (version.major === 3) return version.minor >= 1;
        return version.major > 3;
    }

    if (env.TERM_PROGRAM === "WezTerm") {
        const version = getVersion(env.TERM_PROGRAM_VERSION);
        return version.major >= 20200620;
    }

    if (env.TERM_PROGRAM === "vscode") {
        const version = getVersion(env.TERM_PROGRAM_VERSION);
        return version.major > 1 || (version.major === 1 && version.minor >= 72);
    }

    if (env.VTE_VERSION) {
        if (env.VTE_VERSION === "0.50.0") return false;
        const version = getVersion(env.VTE_VERSION);
        return version.major > 0 || version.minor >= 50;
    }

    return false;
}

function getVersion(versionString: string | undefined): { major: number; minor: number; patch: number } {
    if (!versionString) return { major: 0, minor: 0, patch: 0 };
    if (/^\d{3,4}$/.test(versionString)) {
        const match = /(\d{1,2})(\d{2})/.exec(versionString);
        return {
            major: 0,
            minor: parseInt(match?.[1] || "0", 10),
            patch: parseInt(match?.[2] || "0", 10)
        };
    }
    const parts = versionString.split(".").map(p => parseInt(p, 10));
    return {
        major: parts[0] || 0,
        minor: parts[1] || 0,
        patch: parts[2] || 0
    };
}

/**
 * Global check for terminal hyperlink support.
 */
export function hasLinkSupport(): boolean {
    if (supportsHyperlink(process.stdout as any)) return true;

    const termProgram = process.env.TERM_PROGRAM;
    if (termProgram && ["ghostty", "Hyper", "kitty", "alacritty"].includes(termProgram)) return true;

    if (process.env.TERM?.includes("kitty")) return true;

    return false;
}
