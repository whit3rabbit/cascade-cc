const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const LIBS_PATH = path.join(__dirname, '..', 'libs.json');

function getLatestVersion(packageName) {
    try {
        console.log(`[*] Fetching latest version for ${packageName}...`);
        return execSync(`npm view ${packageName} version`, { encoding: 'utf8' }).trim();
    } catch (err) {
        console.error(`[!] Failed to fetch version for ${packageName}: ${err.message}`);
        return null;
    }
}

function updateLibs() {
    if (!fs.existsSync(LIBS_PATH)) {
        console.error(`[!] libs.json not found at ${LIBS_PATH}`);
        process.exit(1);
    }

    const libs = JSON.parse(fs.readFileSync(LIBS_PATH, 'utf8'));
    const updatedLibs = JSON.parse(JSON.stringify(libs)); // Deep clone
    const latestVersions = {};

    // First pass: find latest versions for all unique packages
    const uniquePackages = [...new Set(libs.map(l => l.name))];
    for (const pkg of uniquePackages) {
        const latest = getLatestVersion(pkg);
        if (latest) {
            latestVersions[pkg] = latest;
        }
    }

    // Second pass: update versions
    const packageGroups = {};
    libs.forEach((lib, index) => {
        if (!packageGroups[lib.name]) packageGroups[lib.name] = [];
        packageGroups[lib.name].push({ ...lib, index });
    });

    for (const pkgName in packageGroups) {
        const group = packageGroups[pkgName];
        const latest = latestVersions[pkgName];
        if (!latest) continue;

        // Filter for versions that are NOT fixed and NOT "latest"
        const updatableEntries = group.filter(l => !l.fixed && l.version !== 'latest');

        if (updatableEntries.length > 0) {
            // Find the highest version among updatable entries
            const highest = updatableEntries.sort((a, b) => {
                // Simple semver compare
                const partsA = a.version.split('.').map(Number);
                const partsB = b.version.split('.').map(Number);
                for (let i = 0; i < 3; i++) {
                    if ((partsA[i] || 0) > (partsB[i] || 0)) return -1;
                    if ((partsA[i] || 0) < (partsB[i] || 0)) return 1;
                }
                return 0;
            })[0];

            if (highest.version !== latest) {
                console.log(`[+] Updating ${pkgName}: ${highest.version} -> ${latest}`);
                updatedLibs[highest.index].version = latest;
            }
        } else {
            console.log(`[~] Skipping update for ${pkgName} (all versions are fixed or "latest")`);
        }
    }

    fs.writeFileSync(LIBS_PATH, JSON.stringify(updatedLibs, null, 4));
    console.log(`\n[COMPLETE] Updated libs.json saved.`);
}

updateLibs();
