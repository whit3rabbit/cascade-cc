import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { createHash } from 'node:crypto';
import { log } from '../logger/loggerService.js';
import { getSettings, updateSettings } from '../terminal/settings.js';
// import { getVersion } from '../../utils/version.js';
const getVersion = () => "2.0.76";
import { SQ } from '../fetch/axiosService.js'; // Assuming SQ is axios or similar wrapper

const logger = log('NativeUpdater');

const CDN_URL = "https://downloads.claude.ai/claude-code-releases";
const GCS_URL = "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases";

export interface UpdateResult {
    success: boolean;
    latestVersion: string | null;
    wasUpdated: boolean;
    lockFailed?: boolean;
    lockHolderPid?: number;
}

function getPlatformId(): string {
    const platform = process.platform;
    const arch = process.arch === 'x64' ? 'x64' : process.arch === 'arm64' ? 'arm64' : null;
    if (!arch) throw new Error(`Unsupported architecture: ${process.arch}`);

    // Simplified version of Ob() from chunk_491.ts
    // In chunk_491, it also checks for musl on linux
    const isMusl = platform === 'linux' && fs.readdirSync('/lib').some(f => f.includes('ld-musl-'));
    if (platform === 'linux' && isMusl) return `linux-${arch}-musl`;

    return `${platform}-${arch}`;
}

function getBinaryName(platformId: string): string {
    return platformId.startsWith('win32') ? 'claude.exe' : 'claude';
}

export function getNativePaths() {
    const platformId = getPlatformId();
    const binaryName = getBinaryName(platformId);

    // Using XDG-like paths as seen in chunk_490/491
    const baseDir = path.join(os.homedir(), '.local', 'share', 'claude');
    const cacheDir = path.join(os.homedir(), '.cache', 'claude');
    const stateDir = path.join(os.homedir(), '.local', 'state', 'claude');
    const binDir = path.join(os.homedir(), '.local', 'bin');

    return {
        versions: path.join(baseDir, 'versions'),
        staging: path.join(cacheDir, 'staging'),
        locks: path.join(stateDir, 'locks'),
        executable: path.join(binDir, binaryName)
    };
}

async function fetchVersion(channel: string): Promise<string> {
    const response = await SQ.get(`${GCS_URL}/${channel}`, {
        timeout: 30000,
        responseType: 'text'
    });
    return response.data.trim();
}

async function downloadAndVerify(url: string, checksum: string, destPath: string): Promise<void> {
    const response = await SQ.get(url, {
        timeout: 300000,
        responseType: 'arraybuffer'
    });
    const hash = createHash('sha256');
    hash.update(response.data);
    const actualChecksum = hash.digest('hex');

    if (actualChecksum !== checksum) {
        throw new Error(`Checksum mismatch: expected ${checksum}, got ${actualChecksum}`);
    }

    fs.writeFileSync(destPath, Buffer.from(response.data));
    fs.chmodSync(destPath, 0o755);
}

async function downloadBinary(version: string, stagingPath: string): Promise<void> {
    if (fs.existsSync(stagingPath)) {
        fs.rmSync(stagingPath, { recursive: true, force: true });
    }
    fs.mkdirSync(stagingPath, { recursive: true });

    const platform = getPlatformId();
    const manifestUrl = `${GCS_URL}/${version}/manifest.json`;
    const manifestResponse = await SQ.get(manifestUrl, { timeout: 10000, responseType: 'json' });
    const manifest = manifestResponse.data;

    const platformInfo = manifest.platforms[platform];
    if (!platformInfo) {
        throw new Error(`Platform ${platform} not found in manifest for version ${version}`);
    }

    const binaryName = getBinaryName(platform);
    const downloadUrl = `${GCS_URL}/${version}/${platform}/${binaryName}`;
    const destPath = path.join(stagingPath, binaryName);

    await downloadAndVerify(downloadUrl, platformInfo.checksum, destPath);
}

export async function checkForNativeUpdate(target: string = 'latest', force: boolean = false): Promise<UpdateResult> {
    try {
        const currentVersion = getVersion();
        const latestVersion = /^v?\d+\.\d+\.\d+(-\S+)?$/.test(target) ? target.replace(/^v/, '') : await fetchVersion(target);

        if (!force && latestVersion === currentVersion) {
            return { success: true, latestVersion, wasUpdated: false };
        }

        const paths = getNativePaths();
        const versionPath = path.join(paths.versions, latestVersion);

        if (!force && fs.existsSync(versionPath)) {
            logger.info(`Version ${latestVersion} already downloaded, updating symlink`);
            await updateExecutable(versionPath);
            return { success: true, latestVersion, wasUpdated: true };
        }

        const stagingPath = path.join(paths.staging, latestVersion);
        await downloadBinary(latestVersion, stagingPath);

        // Atomic install: copy from staging to versions
        if (!fs.existsSync(paths.versions)) fs.mkdirSync(paths.versions, { recursive: true });
        const binaryName = getBinaryName(getPlatformId());
        const stagedBinary = path.join(stagingPath, binaryName);

        // Move to final version path
        fs.copyFileSync(stagedBinary, versionPath);
        fs.chmodSync(versionPath, 0o755);
        fs.rmSync(stagingPath, { recursive: true, force: true });

        await updateExecutable(versionPath);

        return { success: true, latestVersion, wasUpdated: true };
    } catch (e) {
        logger.error('Failed to check for/install native update', e);
        return { success: false, latestVersion: null, wasUpdated: false };
    }
}

async function updateExecutable(versionPath: string) {
    const paths = getNativePaths();
    const binDir = path.dirname(paths.executable);
    if (!fs.existsSync(binDir)) fs.mkdirSync(binDir, { recursive: true });

    if (process.platform === 'win32') {
        // Windows: move old, copy new
        if (fs.existsSync(paths.executable)) {
            const oldPath = `${paths.executable}.old.${Date.now()}`;
            fs.renameSync(paths.executable, oldPath);
            // Ideally we'd cleanup old. executables later
        }
        fs.copyFileSync(versionPath, paths.executable);
    } else {
        // Unix: symlink
        if (fs.existsSync(paths.executable)) {
            fs.unlinkSync(paths.executable);
        }
        fs.symlinkSync(versionPath, paths.executable);
    }
}

export async function runNativeUpdate(target: string = 'latest', showUI: boolean = false, force: boolean = false): Promise<UpdateResult> {
    const settings = getSettings('userSettings');

    if (!force && settings.installMethod !== 'native') {
        return { success: false, latestVersion: null, wasUpdated: false };
    }

    const result = await checkForNativeUpdate(target, force);

    if (result.success && result.wasUpdated) {
        // Update settings in place
        updateSettings('userSettings', {
            installMethod: 'native',
            autoUpdates: false,
            autoUpdatesProtectedForNative: true
        });
    }

    return result;
}

export async function setupLauncher(force: boolean = false): Promise<Array<{ message: string, userActionRequired?: boolean, type: string }>> {
    const paths = getNativePaths();
    const messages: Array<{ message: string, userActionRequired?: boolean, type: string }> = [];

    if (!fs.existsSync(paths.executable)) {
        return messages; // Not installed
    }

    const binDir = path.dirname(paths.executable);
    const envPath = process.env.PATH || '';
    const normalizedEnvPaths = envPath.split(path.delimiter).map(p => path.resolve(p).toLowerCase());
    const normalizedBinDir = path.resolve(binDir).toLowerCase();

    if (!normalizedEnvPaths.includes(normalizedBinDir)) {
        if (process.platform === 'win32') {
            messages.push({
                message: `Native installation exists but ${binDir} is not in your PATH. Add it via Environment Variables and restart your terminal.`,
                userActionRequired: true,
                type: 'path'
            });
        } else {
            const shell = process.env.SHELL || '';
            let rcFile = '~/.bashrc';
            if (shell.includes('zsh')) rcFile = '~/.zshrc';
            else if (shell.includes('fish')) rcFile = '~/.config/fish/config.fish';

            messages.push({
                message: `Native installation exists but ${binDir} is not in your PATH. Run:\necho 'export PATH="$HOME/.local/bin:$PATH"' >> ${rcFile} && source ${rcFile}`,
                userActionRequired: true,
                type: 'path'
            });
        }
    }

    return messages;
}

export async function cleanupOldVersions() {
    const paths = getNativePaths();
    if (!fs.existsSync(paths.versions)) return;

    try {
        const versions = fs.readdirSync(paths.versions)
            .map(v => ({ name: v, path: path.join(paths.versions, v), mtime: fs.statSync(path.join(paths.versions, v)).mtime }))
            .sort((a, b) => b.mtime.getTime() - a.mtime.getTime());

        // Keep last 5 versions
        const toDelete = versions.slice(5);
        for (const v of toDelete) {
            fs.unlinkSync(v.path);
        }
    } catch (e) {
        logger.error('Failed to cleanup old versions', e);
    }
}
