const fs = require('fs');
const path = require('path');

const DEFAULT_KB_PATH = path.resolve('knowledge_base.json');
const CUSTOM_KB_PATH = path.resolve('custom_knowledge_base.json');

function resolveKnowledgeBasePath() {
    if (fs.existsSync(CUSTOM_KB_PATH)) return CUSTOM_KB_PATH;
    if (fs.existsSync(DEFAULT_KB_PATH)) return DEFAULT_KB_PATH;
    return null;
}

function loadKnowledgeBase() {
    const kbPath = resolveKnowledgeBasePath();
    if (!kbPath) return { kb: null, path: null };
    const kb = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
    return { kb, path: kbPath };
}

function buildProjectStructureIndex(projectStructure) {
    if (!projectStructure || typeof projectStructure !== 'object') return null;

    const files = [];
    const dirs = new Map();
    const dirSet = new Set();

    function addDirFile(dirPath, fileName) {
        if (!dirs.has(dirPath)) dirs.set(dirPath, new Set());
        dirs.get(dirPath).add(fileName);
    }

    function walk(node, segments) {
        if (!node || typeof node !== 'object') return;
        const filesList = Array.isArray(node.files) ? node.files : [];
        const dirPath = segments.join('/');
        if (filesList.length) {
            dirSet.add(dirPath);
            filesList.forEach(file => {
                files.push(dirPath ? `${dirPath}/${file}` : file);
                addDirFile(dirPath, file);
            });
        }
        Object.entries(node).forEach(([key, value]) => {
            if (key === 'files' || key === 'description') return;
            if (value && typeof value === 'object') walk(value, segments.concat(key));
        });
    }

    Object.entries(projectStructure).forEach(([rootKey, value]) => {
        walk(value, [rootKey]);
    });

    const byBase = new Map();
    files.forEach(p => {
        const base = path.posix.basename(p);
        if (!byBase.has(base)) byBase.set(base, []);
        byBase.get(base).push(p);
    });

    const topLevelDirs = new Set();
    files.forEach(p => {
        const parts = p.split('/');
        if (parts.length >= 2) {
            topLevelDirs.add(`${parts[0]}/${parts[1]}`);
        }
    });

    return {
        files,
        dirs,
        byBase,
        topLevelDirs: Array.from(topLevelDirs).sort()
    };
}

function buildProjectStructureSlice(index, options = {}) {
    if (!index) return null;

    const {
        relDir,
        neighborPaths = [],
        importHints = [],
        maxDirs = 12,
        maxFilesPerDir = 60,
        maxRelated = 80,
        maxTopLevel = 60
    } = options;

    const normalize = p => p.replace(/\\/g, '/');
    const candidateDirs = new Set();
    const candidateBases = new Set();

    function addDirFromPath(p) {
        if (!p) return;
        const dir = path.posix.dirname(normalize(p));
        if (dir && dir !== '.') candidateDirs.add(dir);
    }

    if (relDir && relDir !== '.') {
        candidateDirs.add(normalize(relDir));
        const parent = path.posix.dirname(normalize(relDir));
        if (parent && parent !== '.') candidateDirs.add(parent);
    }

    neighborPaths.forEach(p => {
        if (!p) return;
        candidateBases.add(path.posix.basename(normalize(p)));
        addDirFromPath(p);
    });

    importHints.forEach(hint => {
        const suggestion = typeof hint === 'string' ? hint : hint?.suggestion;
        if (!suggestion) return;
        const normalized = normalize(suggestion);
        candidateBases.add(path.posix.basename(normalized));
        if (normalized.startsWith('.')) {
            if (relDir && relDir !== '.') {
                const absPath = path.posix.normalize(path.posix.join(relDir, normalized));
                addDirFromPath(absPath);
            }
        } else if (normalized.startsWith('/')) {
            addDirFromPath(normalized.slice(1));
        } else {
            addDirFromPath(normalized);
        }
    });

    function findClosestDir(dir) {
        let current = dir;
        while (current && current !== '.' && current !== '/') {
            if (index.dirs.has(current)) return current;
            const parent = path.posix.dirname(current);
            if (parent === current) break;
            current = parent;
        }
        return null;
    }

    const selectedDirs = [];
    for (const dir of candidateDirs) {
        const resolved = findClosestDir(dir);
        if (resolved && !selectedDirs.includes(resolved)) selectedDirs.push(resolved);
    }

    const directories = {};
    selectedDirs.slice(0, maxDirs).forEach(dir => {
        const entries = index.dirs.get(dir);
        if (!entries || entries.size === 0) return;
        directories[dir] = Array.from(entries).sort().slice(0, maxFilesPerDir);
    });

    const relatedFiles = [];
    for (const base of candidateBases) {
        const matches = index.byBase.get(base);
        if (matches) {
            matches.forEach(m => {
                if (relatedFiles.length < maxRelated) relatedFiles.push(m);
            });
        }
        if (relatedFiles.length >= maxRelated) break;
    }

    return {
        topLevelDirs: index.topLevelDirs.slice(0, maxTopLevel),
        directories,
        relatedFiles
    };
}

module.exports = {
    loadKnowledgeBase,
    resolveKnowledgeBasePath,
    DEFAULT_KB_PATH,
    CUSTOM_KB_PATH,
    buildProjectStructureIndex,
    buildProjectStructureSlice
};
