
// Logic from chunk_490.ts (Updater Service, Process Lock)

// --- Platform and Paths (Ob, _b) ---
export function getPlatformSystem() {
    const platform = process.platform;
    const arch = process.arch === "x64" ? "x64" : process.arch === "arm64" ? "arm64" : "unknown";
    return `${platform}-${arch}`;
}

export function getInstallationLayout() {
    const platform = getPlatformSystem();
    const exeName = platform.startsWith("win32") ? "claude.exe" : "claude";
    return {
        versions: require("path").join(getXdgStateHome(), "claude", "versions"),
        staging: require("path").join(getXdgStateHome(), "claude", "staging"),
        locks: require("path").join(getXdgStateHome(), "claude", "locks"),
        executable: require("path").join(getXdgStateHome(), "claude", exeName)
    };
}

// --- Maintenance and Locking (zW1, A$0) ---
export async function performMaintenance() {
    const layout = getInstallationLayout();
    // Stub for cleanup logic
    console.log("Cleaning up old versions and locks...");
}

export async function acquireVersionLock(version: string, operation: () => Promise<void>) {
    const layout = getInstallationLayout();
    const lockPath = require("path").join(layout.locks, `${version}.lock`);
    console.log(`Locking version ${version}...`);
    try {
        await operation();
    } finally {
        console.log(`Unlocked version ${version}`);
    }
}

// --- XDG & Shell Helpers ---
export function getXdgStateHome() {
    return process.env.XDG_STATE_HOME || require("path").join(require("os").homedir(), ".local", "state");
}

export function getShellConfigs() {
    const home = require("os").homedir();
    return {
        zsh: require("path").join(home, ".zshrc"),
        bash: require("path").join(home, ".bashrc")
    };
}
